"""
================================================================================
 POTHOLE CNN CLASSIFIER
 Image-only pothole detector built on a ConvNeXt-Tiny backbone.

 This version removes the IMU fusion path and trains on image-level labels:
     data/
         images/          ← JPG/PNG files
         labels.csv       ← columns: image_file,label
     checkpoints/       ← saved model weights
     logs/              ← TensorBoard logs

 Supported labels:
     0 / 1, false / true, no_pothole / pothole, negative / positive
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS & GLOBAL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
import os, sys, math, time, logging, warnings, random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast

from PIL import Image
import albumentations as A           # pip install albumentations
from albumentations.pytorch import ToTensorV2
import torchvision.models as tv_models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False
    warnings.warn("TensorBoard not found; logging to console only.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  HYPER-PARAMETER DATACLASS  (single source of truth)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Paths
    data_dir:        str  = "data"
    image_dir:       str  = "data/images"
    labels_csv:      str  = "data/labels.csv"
    checkpoint_dir:  str  = "checkpoints"
    log_dir:         str  = "logs"

    # Vision backbone
    vision_model:    str  = "resnet18"   # local torchvision backbone
    img_size:        int  = 256
    freeze_backbone: bool = False     # fine-tune entire backbone
    threshold:       float = 0.5

    # Architecture dims
    vision_embed_dim: int = 256
    dropout:          float = 0.2

    # Training
    epochs:          int   = 30
    batch_size:      int   = 16
    lr:              float = 2e-4
    weight_decay:    float = 1e-4
    grad_clip:       float = 1.0
    warmup_epochs:   int   = 5
    T0:              int   = 10       # CosineAnnealingWarmRestarts T_0
    T_mult:          int   = 2

    # Misc
    seed:            int   = 42
    num_workers:     int   = 0
    pin_memory:      bool  = True
    use_amp:         bool  = True     # automatic mixed precision
    val_split:       float = 0.15
    test_split:      float = 0.10
    class_names:     Tuple[str, str] = ("NO_POTHOLE", "POTHOLE")


CFG = Config()


# ──────────────────────────────────────────────────────────────────────────────
# 2.  REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything(CFG.seed)


def get_best_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_supports_amp(device: torch.device) -> bool:
    return device.type == "cuda"


# ──────────────────────────────────────────────────────────────────────────────
# 3.  IMU FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
class IMUFeatureExtractor:
    """
    Converts a raw IMU window (N × 6) into a rich feature vector.

    Features extracted:
      • Statistical: mean, std, min, max, range, kurtosis, skewness  (per axis)
      • Jerk-domain: peak-to-peak jerk magnitude on Z axis
      • Energy: RMS of combined acceleration vector
      • Spectral: FFT power spectrum of Z-axis acceleration (top-k bins)
      • Cross-axis correlation: ax-ay, ax-az, ay-az
      • Sliding-window variance trajectory (variance over sub-windows)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scaler: Optional[RobustScaler] = None

    # ------------------------------------------------------------------
    def _stat_features(self, window: np.ndarray) -> np.ndarray:
        """window: (T, 6)  → ax,ay,az,gx,gy,gz"""
        feats = []
        for col in range(window.shape[1]):
            x = window[:, col]
            feats += [
                x.mean(), x.std(), x.min(), x.max(),
                x.max() - x.min(),                           # range
                float(pd.Series(x).kurtosis()),
                float(pd.Series(x).skew()),
            ]
        return np.array(feats, dtype=np.float32)             # 6 × 7 = 42

    # ------------------------------------------------------------------
    def _jerk_features(self, window: np.ndarray) -> np.ndarray:
        """Peak-to-peak jerk on Z axis (index 2)."""
        az  = window[:, 2]
        jerk = np.diff(az)
        return np.array([
            jerk.max() - jerk.min(),   # peak-to-peak jerk magnitude
            np.abs(jerk).max(),        # max absolute jerk
            np.abs(jerk).mean(),       # mean absolute jerk
        ], dtype=np.float32)           # 3

    # ------------------------------------------------------------------
    def _energy_features(self, window: np.ndarray) -> np.ndarray:
        """RMS of resultant acceleration and gyroscope vectors."""
        accel_mag = np.sqrt((window[:, :3] ** 2).sum(axis=1))
        gyro_mag  = np.sqrt((window[:, 3:] ** 2).sum(axis=1))
        return np.array([
            np.sqrt((accel_mag ** 2).mean()),   # accel RMS
            np.sqrt((gyro_mag  ** 2).mean()),   # gyro  RMS
            accel_mag.max(),
            gyro_mag.max(),
        ], dtype=np.float32)                    # 4

    # ------------------------------------------------------------------
    def _spectral_features(self, window: np.ndarray) -> np.ndarray:
        """FFT power spectrum of Z-axis acceleration."""
        az   = window[:, 2]
        N    = len(az)
        win  = np.hanning(N)
        yf   = np.abs(fft(az * win)[:N // 2]) ** 2
        # Resample to fixed number of bins
        bins = self.cfg.fft_bins
        if len(yf) >= bins:
            yf_resampled = yf[:bins]
        else:
            yf_resampled = np.pad(yf, (0, bins - len(yf)))
        yf_resampled = yf_resampled / (yf_resampled.sum() + 1e-8)   # normalize
        return yf_resampled.astype(np.float32)                       # fft_bins

    # ------------------------------------------------------------------
    def _cross_correlation(self, window: np.ndarray) -> np.ndarray:
        """Pearson correlation between axis pairs."""
        ax, ay, az = window[:, 0], window[:, 1], window[:, 2]
        def safe_corr(a, b):
            denom = np.std(a) * np.std(b)
            return float(np.corrcoef(a, b)[0, 1]) if denom > 1e-8 else 0.0
        return np.array([
            safe_corr(ax, ay),
            safe_corr(ax, az),
            safe_corr(ay, az),
        ], dtype=np.float32)                                         # 3

    # ------------------------------------------------------------------
    def _sliding_variance(self, window: np.ndarray, n_sub: int = 5) -> np.ndarray:
        """Variance trajectory: variance of Z-axis in n_sub sub-windows."""
        az     = window[:, 2]
        chunks = np.array_split(az, n_sub)
        return np.array([c.var() for c in chunks], dtype=np.float32) # n_sub

    # ------------------------------------------------------------------
    def extract(self, window: np.ndarray) -> np.ndarray:
        """
        window: np.ndarray of shape (T, 6) – raw IMU values
        returns: 1-D feature vector of fixed length
        """
        if window.shape[0] < 4:
            # Pad if window is too short (edge case)
            pad = np.zeros((4 - window.shape[0], window.shape[1]))
            window = np.vstack([window, pad])

        feats = np.concatenate([
            self._stat_features(window),         # 42
            self._jerk_features(window),         # 3
            self._energy_features(window),       # 4
            self._spectral_features(window),     # fft_bins (32)
            self._cross_correlation(window),     # 3
            self._sliding_variance(window),      # 5
        ])
        return feats                             # total: 42+3+4+32+3+5 = 89

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        return 42 + 3 + 4 + self.cfg.fft_bins + 3 + 5


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ALBUMENTATIONS AUGMENTATION PIPELINES
# ──────────────────────────────────────────────────────────────────────────────
def build_train_transform(img_size: int) -> A.Compose:
    """
    Road-environment-specific augmentations:
      • Brightness/contrast shifts   → dawn/dusk/night/overcast
      • Random rain overlay          → wet roads
      • Motion blur                  → vehicle vibration
      • Coarse dropout               → sensor occlusion/dirt
      • Perspective distortion       → camera mounting angle variance
      • Gaussian noise               → cheap CMOS sensor noise
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
        A.RandomRain(rain_type="default", p=0.1),
        A.MotionBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(p=0.2),
        A.Perspective(scale=(0.02, 0.05), p=0.2),
        A.CoarseDropout(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def build_val_transform(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# 5.  SENSOR NOISE INJECTION  (for IMU augmentation)
# ──────────────────────────────────────────────────────────────────────────────
def inject_sensor_noise(
    imu_features: torch.Tensor,
    noise_std: float = 0.02,
    dropout_prob: float = 0.05,
) -> torch.Tensor:
    """
    Simulates real-world IMU sensor degradation:
      • Additive Gaussian noise (quantization, thermal)
      • Random feature dropout (packet loss, sensor glitch)
    Applied only during training.
    """
    noise  = torch.randn_like(imu_features) * noise_std
    mask   = (torch.rand_like(imu_features) > dropout_prob).float()
    return (imu_features + noise) * mask


# ──────────────────────────────────────────────────────────────────────────────
# 6.  CUSTOM DATASET
# ──────────────────────────────────────────────────────────────────────────────
def normalize_binary_label(value) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "positive", "pothole", "damaged"}:
            return 1
        if normalized in {"0", "false", "no", "negative", "no_pothole", "clear", "normal"}:
            return 0
        try:
            return int(float(normalized) > 0.5)
        except ValueError:
            raise ValueError(f"Unsupported label value: {value}")
    return int(float(value) > 0.5)


def infer_label_from_path(path: Path) -> Optional[int]:
    parts = {part.lower() for part in path.parts}
    if {"pothole", "positive", "damaged"} & parts:
        return 1
    if {"no_pothole", "normal", "negative", "clear"} & parts:
        return 0
    return None


class PotholeDataset(Dataset):
    """Image-only pothole dataset for binary classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform,
        cfg: Config,
        is_train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.cfg = cfg
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, row: pd.Series) -> Path:
        candidates: List[Path] = []
        for key in ("image_file", "image", "filename", "path"):
            value = row.get(key, None)
            if isinstance(value, str) and value.strip():
                candidate = Path(value)
                candidates.append(candidate if candidate.is_absolute() else self.image_dir / candidate)
                candidates.append(self.image_dir / value)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not resolve image for row {row.name}. Tried: {', '.join(str(p) for p in candidates[:3])}"
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row)
        img = np.array(Image.open(img_path).convert("RGB"))
        image = self.transform(image=img)["image"]

        label_value = row.get("label", row.get("severity_label", row.get("target", 0)))
        label = torch.tensor(normalize_binary_label(label_value), dtype=torch.float32)

        return {
            "image": image,
            "label": label,
            "image_path": str(img_path),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 7.  VISION BACKBONE  (ConvNeXt-Tiny, ImageNet pre-trained)
# ──────────────────────────────────────────────────────────────────────────────
class VisionEncoder(nn.Module):
    """
    Local torchvision ResNet18 backbone with a projection head.

    This avoids any external weight download and keeps the entire model local.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.vision_model != "resnet18":
            log.warning("vision_model=%s is ignored; using local resnet18 backbone", cfg.vision_model)

        self.backbone = tv_models.resnet18(weights=None)
        backbone_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        if not cfg.freeze_backbone:
            # Unfreeze all — fine-tune end-to-end
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            # Freeze the backbone and train only the projection head
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, cfg.vision_embed_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.vision_embed_dim * 2, cfg.vision_embed_dim),
            nn.LayerNorm(cfg.vision_embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W)  →  (B, vision_embed_dim)"""
        feat = self.backbone(x)    # (B, backbone_dim)
        return self.proj(feat)     # (B, vision_embed_dim)


# ──────────────────────────────────────────────────────────────────────────────
# 8.  IMU ENCODER  (MLP with residual connections)
# ──────────────────────────────────────────────────────────────────────────────
class IMUEncoder(nn.Module):
    """
    Two-layer residual MLP that maps the engineered IMU feature vector to
    a latent embedding compatible with the fusion module.
    """

    def __init__(self, input_dim: int, cfg: Config):
        super().__init__()
        hidden = cfg.imu_embed_dim * 2

        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, cfg.imu_embed_dim)
        self.res  = nn.Linear(input_dim, cfg.imu_embed_dim)   # skip connection
        self.act  = nn.GELU()
        self.drop = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(cfg.imu_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D)  →  (B, imu_embed_dim)"""
        h  = self.act(self.fc1(x))
        h  = self.drop(h)
        h  = self.fc2(h)
        return self.norm(h + self.res(x))


# ──────────────────────────────────────────────────────────────────────────────
# 9.  CROSS-ATTENTION FUSION MODULE
# ──────────────────────────────────────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion:

      1. Vision attends to IMU   → visual context enriched with kinematic cues
      2. IMU attends to Vision   → kinematic context enriched with structural cues
      3. Concatenate & project   → fused representation

    This is superior to naive concatenation because:
      • The model learns WHICH visual regions correlate to high-G events
      • IMU features can dynamically weight visual texture importance
      • Attention is learnable and interpretable
    """

    def __init__(self, cfg: Config):
        super().__init__()
        v_dim = cfg.vision_embed_dim
        i_dim = cfg.imu_embed_dim
        f_dim = cfg.fusion_dim

        # Project both streams to the same fusion dimension
        self.v_proj = nn.Linear(v_dim, f_dim)
        self.i_proj = nn.Linear(i_dim, f_dim)

        # Multi-head cross-attention (vision ← IMU)
        self.v2i_attn = nn.MultiheadAttention(
            embed_dim=f_dim, num_heads=cfg.num_heads,
            dropout=cfg.dropout, batch_first=True,
        )
        # Multi-head cross-attention (IMU ← vision)
        self.i2v_attn = nn.MultiheadAttention(
            embed_dim=f_dim, num_heads=cfg.num_heads,
            dropout=cfg.dropout, batch_first=True,
        )

        self.norm_v  = nn.LayerNorm(f_dim)
        self.norm_i  = nn.LayerNorm(f_dim)
        self.drop    = nn.Dropout(cfg.dropout)

        # Final fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(f_dim * 2, f_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(f_dim, f_dim),
            nn.LayerNorm(f_dim),
        )

    def forward(
        self,
        v: torch.Tensor,   # (B, vision_embed_dim)
        i: torch.Tensor,   # (B, imu_embed_dim)
    ) -> torch.Tensor:
        # Project to common fusion dimension and add sequence dim
        vp = self.v_proj(v).unsqueeze(1)   # (B, 1, F)
        ip = self.i_proj(i).unsqueeze(1)   # (B, 1, F)

        # Vision attends to IMU context
        v_ctx, _ = self.v2i_attn(query=vp, key=ip, value=ip)
        v_fused  = self.norm_v(vp + self.drop(v_ctx)).squeeze(1)  # (B, F)

        # IMU attends to visual context
        i_ctx, _ = self.i2v_attn(query=ip, key=vp, value=vp)
        i_fused  = self.norm_i(ip + self.drop(i_ctx)).squeeze(1)  # (B, F)

        # Concatenate and project to final embedding
        combined = torch.cat([v_fused, i_fused], dim=-1)          # (B, 2F)
        return self.fusion_mlp(combined)                           # (B, F)


# ──────────────────────────────────────────────────────────────────────────────
# 10. FULL IMAGE-ONLY POTHOLE NETWORK
# ──────────────────────────────────────────────────────────────────────────────
class PotholeNet(nn.Module):
    """ConvNeXt-Tiny image classifier for pothole detection."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.vision_enc = VisionEncoder(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.vision_embed_dim, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.vision_enc(image)
        logits = self.classifier(features).squeeze(-1)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# 11. LOSS FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def build_loss(pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    if pos_weight is not None:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss()


# ──────────────────────────────────────────────────────────────────────────────
# 12. METRICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(logits: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int32)
    targets = targets.astype(np.int32)

    metrics = {
        "Accuracy": accuracy_score(targets, preds),
        "Precision": precision_score(targets, preds, zero_division=0),
        "Recall": recall_score(targets, preds, zero_division=0),
        "F1": f1_score(targets, preds, zero_division=0),
    }
    try:
        metrics["AUC"] = roc_auc_score(targets, probs)
    except ValueError:
        metrics["AUC"] = float("nan")
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 13. WEIGHTED SAMPLER  (handles class imbalance for rare severe potholes)
# ──────────────────────────────────────────────────────────────────────────────
def build_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Over-sample the minority class for binary classification."""
    labels = labels.astype(np.int32)
    class_counts = np.bincount(labels, minlength=2).astype(np.float32)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(weights),
        replacement=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 14. DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_data(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader, float]:
    csv_path = Path(cfg.labels_csv)
    image_dir = Path(cfg.image_dir)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "image_file" not in df.columns or "label" not in df.columns:
            raise ValueError("labels.csv must contain image_file and label columns")
        df = df.copy()
        df["label"] = df["label"].apply(normalize_binary_label)
    else:
        records = []
        for folder in sorted(p for p in image_dir.iterdir() if p.is_dir()):
            label = infer_label_from_path(folder)
            if label is None:
                continue
            for image_path in sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg")) + sorted(folder.glob("*.png")):
                records.append({"image_file": str(image_path.relative_to(image_dir)), "label": label})
        if not records:
            raise FileNotFoundError(
                f"No labels CSV found at {csv_path} and no labeled subfolders found under {image_dir}. "
                "Create a labels.csv with columns image_file,label or organize images into pothole/no_pothole folders."
            )
        df = pd.DataFrame(records)

    if len(df) < 3:
        raise ValueError("Need at least 3 labeled samples to build train/val/test splits")

    stratify = df["label"] if df["label"].nunique() > 1 else None
    df_train_val, df_test = train_test_split(
        df, test_size=cfg.test_split, random_state=cfg.seed, shuffle=True, stratify=stratify
    )
    stratify_train = df_train_val["label"] if df_train_val["label"].nunique() > 1 else None
    val_ratio = cfg.val_split / (1 - cfg.test_split)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_ratio,
        random_state=cfg.seed,
        shuffle=True,
        stratify=stratify_train,
    )

    log.info(f"Dataset split — Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    t_transform = build_train_transform(cfg.img_size)
    v_transform = build_val_transform(cfg.img_size)

    ds_train = PotholeDataset(df_train, cfg.image_dir, t_transform, cfg, is_train=True)
    ds_val = PotholeDataset(df_val, cfg.image_dir, v_transform, cfg, is_train=False)
    ds_test = PotholeDataset(df_test, cfg.image_dir, v_transform, cfg, is_train=False)

    sampler = build_weighted_sampler(df_train["label"].to_numpy())

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    pos = max(int(df_train["label"].sum()), 1)
    neg = max(int((1 - df_train["label"]).sum()), 1)
    pos_weight = float(neg / pos)
    return dl_train, dl_val, dl_test, pos_weight


# ──────────────────────────────────────────────────────────────────────────────
# 15. TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler:    GradScaler,
    device:    torch.device,
    cfg:       Config,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_logits: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    n = 0
    amp_enabled = cfg.use_amp and device.type == "cuda"

    for batch in loader:
        img  = batch["image"].to(device, non_blocking=True)
        tgt  = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            logits = model(img)
            loss = criterion(logits, tgt)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs         = tgt.size(0)
        total_loss += loss.item() * bs
        n          += bs
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(tgt.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits)
    targets_np = np.concatenate(all_targets)
    metrics = compute_metrics(logits_np, targets_np, threshold=cfg.threshold)
    metrics["loss"] = total_loss / n
    return metrics


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    cfg:       Config,
) -> Dict[str, float]:
    model.eval()
    total_loss, n = 0.0, 0
    all_logits, all_tgts = [], []
    amp_enabled = cfg.use_amp and device.type == "cuda"

    for batch in loader:
        img  = batch["image"].to(device, non_blocking=True)
        tgt  = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            logits = model(img)
            loss = criterion(logits, tgt)

        bs         = tgt.size(0)
        total_loss += loss.item() * bs
        n          += bs
        all_logits.append(logits.cpu().numpy())
        all_tgts.append(tgt.cpu().numpy())

    preds   = np.concatenate(all_logits)
    targets = np.concatenate(all_tgts)
    metrics = compute_metrics(preds, targets)
    metrics["loss"] = total_loss / n
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 16. WARMUP SCHEDULER WRAPPER
# ──────────────────────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_epochs`, then CosineAnnealingWarmRestarts.
    """

    def __init__(self, optimizer, cfg: Config):
        self.optimizer     = optimizer
        self.warmup_epochs = cfg.warmup_epochs
        self.base_lrs      = [pg["lr"] for pg in optimizer.param_groups]
        self.cosine        = CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.T0, T_mult=cfg.T_mult, eta_min=1e-6
        )
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            factor = self.epoch / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * factor
        else:
            self.cosine.step(self.epoch - self.warmup_epochs)

    def get_last_lr(self) -> List[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ──────────────────────────────────────────────────────────────────────────────
# 17. MAIN TRAINING FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def train(cfg: Config = CFG):
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    device = get_best_device()
    log.info(
        "Apple MPS detected: %s | Using device: %s | AMP: %s",
        "yes" if device.type == "mps" else "no",
        device,
        "enabled" if device_supports_amp(device) and cfg.use_amp else "disabled",
    )

    # ── Data ──────────────────────────────────────────────────────────
    dl_train, dl_val, dl_test, pos_weight = load_data(cfg)

    # ── Model ─────────────────────────────────────────────────────────
    model = PotholeNet(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = build_loss(pos_weight=pos_weight_tensor)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "vision_enc.backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr * 0.1},
            {"params": head_params, "lr": cfg.lr},
        ],
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )
    
    scheduler    = WarmupCosineScheduler(optimizer, cfg)
    amp_scaler   = GradScaler(enabled=device.type == "cuda" and cfg.use_amp)
    early_stop   = EarlyStopping(patience=150)

    writer = SummaryWriter(cfg.log_dir) if HAS_TB else None

    best_val_f1   = float("-inf")
    best_ckpt     = Path(cfg.checkpoint_dir) / "best_model.pt"

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(model, dl_train, optimizer, criterion,
                                  amp_scaler, device, cfg)
        val_m   = evaluate(model, dl_val, criterion, device, cfg)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        log.info(
            f"Epoch {epoch:3d}/{cfg.epochs} | "
            f"Train Loss={train_m['loss']:.4f} Acc={train_m['Accuracy']:.3f} F1={train_m['F1']:.3f} | "
            f"Val   Loss={val_m['loss']:.4f}  Acc={val_m['Accuracy']:.3f}  F1={val_m['F1']:.3f}  "
            f"Prec={val_m['Precision']:.3f} Rec={val_m['Recall']:.3f} AUC={val_m['AUC']:.3f} | "
            f"LR={lr_now:.2e} | {elapsed:.1f}s"
        )

        # TensorBoard
        if writer:
            writer.add_scalars("Loss",    {"train": train_m["loss"],  "val": val_m["loss"]}, epoch)
            writer.add_scalars("Accuracy", {"train": train_m["Accuracy"], "val": val_m["Accuracy"]}, epoch)
            writer.add_scalars("F1",       {"train": train_m["F1"],       "val": val_m["F1"]}, epoch)
            writer.add_scalar("Val/Precision", val_m["Precision"], epoch)
            writer.add_scalar("Val/Recall",    val_m["Recall"],    epoch)
            writer.add_scalar("Val/AUC",       val_m["AUC"],       epoch)
            writer.add_scalar("LR",               lr_now,                  epoch)

        # Save best checkpoint
        if val_m["F1"] > best_val_f1:
            best_val_f1 = val_m["F1"]
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1":         best_val_f1,
                "cfg":            cfg,
            }, best_ckpt)
            log.info(f"  ✓ New best checkpoint saved (Val F1={best_val_f1:.4f})")

        if early_stop(-val_m["F1"]):
            log.info(f"Early stopping triggered at epoch {epoch}")
            break

    if writer:
        writer.close()

    # ── Final test evaluation ──────────────────────────────────────────
    log.info("\n── Final Test Evaluation ──")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(model, dl_test, criterion, device, cfg)
    log.info(
        f"Test Acc={test_m['Accuracy']:.3f} | Prec={test_m['Precision']:.3f} | "
        f"Rec={test_m['Recall']:.3f} | F1={test_m['F1']:.3f} | AUC={test_m['AUC']:.3f}"
    )

    return model, test_m


# ──────────────────────────────────────────────────────────────────────────────
# 18. INFERENCE  (production scoring)
# ──────────────────────────────────────────────────────────────────────────────
class PotholeScorer:
    """Production-ready image-only pothole classifier."""

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        if device is None:
            device = get_best_device().type
        self.device = torch.device(device)
        # Some checkpoints were saved when this file was executed as __main__.
        # That causes pickle to look for classes (like Config) under module '__main__',
        # which breaks when importing this module. Workaround: temporarily map
        # '__main__' to this module so unpickling can resolve the dataclass.
        import sys

        orig_main = sys.modules.get("__main__")
        sys.modules["__main__"] = sys.modules.get(__name__)
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        finally:
            # restore previous __main__ mapping
            if orig_main is not None:
                sys.modules["__main__"] = orig_main
            else:
                try:
                    del sys.modules["__main__"]
                except KeyError:
                    pass
        cfg  = ckpt["cfg"]
        self.model = PotholeNet(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.transform = build_val_transform(cfg.img_size)
        self.threshold = getattr(cfg, "threshold", 0.5)

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        threshold: Optional[float] = None,
    ) -> Dict:
        img = np.array(Image.open(image_path).convert("RGB"))
        img_t = self.transform(image=img)["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(img_t)
            prob = torch.sigmoid(logit).item()

        thr = self.threshold if threshold is None else threshold
        pred = int(prob >= thr)
        return {
            "probability": round(prob, 4),
            "threshold": round(thr, 4),
            "label": "POTHOLE" if pred else "NO_POTHOLE",
            "predicted_class": pred,
        }

    @torch.no_grad()
    def predict_batch(
        self,
        image_paths: List[str],
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        return [self.predict(p, threshold=threshold) for p in image_paths]


# ──────────────────────────────────────────────────────────────────────────────
# 19. SYNTHETIC DATA GENERATOR  (for smoke-testing without real hardware)
# ──────────────────────────────────────────────────────────────────────────────
def generate_synthetic_dataset(
    n_samples:  int   = 500,
    image_dir:  str   = "data/images",
    csv_path:   str   = "data/labels.csv",
    seed:       int   = 42,
):
    """
    Creates dummy JPG images and a binary labels CSV so you can smoke-test
    the CNN training pipeline without real annotations.
    """
    import os
    os.makedirs(image_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    rows = []
    for i in range(n_samples):
        label = int(rng.random() > 0.55)
        rows.append({
            "image_file": f"{i:04d}.jpg",
            "label": label,
        })

        img_np = rng.integers(60, 180, size=(128, 128, 3), dtype=np.uint8)
        if label == 1:
            for _ in range(rng.integers(3, 8)):
                y = rng.integers(20, 110)
                thickness = rng.integers(2, 5)
                img_np[y:y + thickness, :, :] = rng.integers(10, 50, size=(thickness, 128, 3), dtype=np.uint8)
            for _ in range(rng.integers(1, 4)):
                x = rng.integers(15, 110)
                y = rng.integers(30, 95)
                radius = rng.integers(5, 18)
                yy, xx = np.ogrid[:128, :128]
                mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
                img_np[mask] = rng.integers(20, 70, size=3, dtype=np.uint8)
        Image.fromarray(img_np).save(f"{image_dir}/{i:04d}.jpg")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    log.info(f"Synthetic dataset: {n_samples} samples written to {csv_path}")
    log.info(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 20. ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image-only pothole classifier")
    parser.add_argument("--mode",     type=str, default="train",
                        choices=["train", "infer", "synth"],
                        help="'train': run training | 'infer': single inference | 'synth': generate synthetic data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--image",      type=str, default=None)
    parser.add_argument("--labels_csv", type=str, default=CFG.labels_csv)
    parser.add_argument("--image_dir", type=str, default=CFG.image_dir)
    parser.add_argument("--threshold", type=float, default=CFG.threshold)
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--synth_n",   type=int, default=500)
    args = parser.parse_args()

    CFG.labels_csv = args.labels_csv
    CFG.image_dir = args.image_dir
    CFG.threshold = args.threshold
    CFG.epochs = args.epochs

    if args.mode == "synth":
        generate_synthetic_dataset(n_samples=args.synth_n, image_dir=args.image_dir, csv_path=args.labels_csv)
        log.info("Synthetic dataset generated. Run with --mode train to start training.")

    elif args.mode == "train":
        if not Path(CFG.labels_csv).exists():
            log.warning("No labels.csv found. Generating synthetic dataset for demo...")
            generate_synthetic_dataset(n_samples=500, image_dir=CFG.image_dir, csv_path=CFG.labels_csv)
        train(CFG)

    elif args.mode == "infer":
        if not Path(args.checkpoint).exists():
            log.error(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

        scorer = PotholeScorer(args.checkpoint)
        if args.image is None:
            raise SystemExit("--image is required in infer mode")
        result   = scorer.predict(str(args.image), threshold=args.threshold)

        print("\n" + "="*50)
        print(f"  POTHOLE PROBABILITY:     {result['probability']:.4f}")
        print(f"  THRESHOLD:               {result['threshold']:.2f}")
        print(f"  CLASS:                   {result['label']}")
        print("="*50 + "\n")
