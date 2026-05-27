"""
================================================================================
 MULTIMODAL POTHOLE SEVERITY SCORING SYSTEM
 Fusing ESP32-CAM Vision + IMU Telemetry via Cross-Attention
 Architecture: ConvNeXt-Tiny + Temporal IMU Encoder + Cross-Attention Fusion
 Output: Continuous severity score in [1.0, 10.0]
================================================================================

DIRECTORY STRUCTURE EXPECTED:
  data/
    images/          ← JPG files, named as <timestamp>.jpg or <index>.jpg
    telemetry.csv    ← columns: timestamp, ax, ay, az, gx, gy, gz
  checkpoints/       ← saved model weights
  logs/              ← TensorBoard logs

CSV FORMAT:
  timestamp,ax,ay,az,gx,gy,gz,severity_label
  (severity_label is optional at inference; required for training)

IMAGE FILENAME FORMATS SUPPORTED:
  - <unix_timestamp>.jpg   e.g.  1718000123456.jpg
  - <index>.jpg            e.g.  0042.jpg
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

import timm                          # pip install timm
from PIL import Image
import albumentations as A           # pip install albumentations
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
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
    image_dir:       str  = r"C:\Users\E028.28\Downloads\archive (2)\images"
    telemetry_csv:   str  = "data/telemetry.csv"
    checkpoint_dir:  str  = "checkpoints"
    log_dir:         str  = "logs"

    # IMU windowing
    imu_window_size: int  = 50        # samples around the trigger event
    imu_step:        int  = 1         # stride for sliding window features
    fft_bins:        int  = 32        # FFT frequency bins to keep

    # Vision backbone
    vision_model:    str  = "convnext_tiny.in12k_ft_in1k"   # timm model name
    img_size:        int  = 256
    freeze_backbone: bool = False     # fine-tune entire backbone

    # Architecture dims
    vision_embed_dim: int = 256
    imu_embed_dim:    int = 128
    fusion_dim:       int = 256
    num_heads:        int = 8
    dropout:          float = 0.2

    # Training
    epochs:          int   = 150
    batch_size:      int   = 16
    lr:              float = 2e-4
    weight_decay:    float = 1e-4
    grad_clip:       float = 1.0
    warmup_epochs:   int   = 5
    T0:              int   = 10       # CosineAnnealingWarmRestarts T_0
    T_mult:          int   = 2

    # Loss
    loss_type:       str   = "wing"   # "wing" | "huber" | "focal_mse"
    wing_w:          float = 10.0
    wing_epsilon:    float = 2.0
    huber_delta:     float = 1.0

    # Misc
    seed:            int   = 42
    num_workers:     int   = 0
    pin_memory:      bool  = True
    use_amp:         bool  = True     # automatic mixed precision
    val_split:       float = 0.15
    test_split:      float = 0.10

    # Severity range
    score_min:       float = 1.0
    score_max:       float = 10.0


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
class PotholeDataset(Dataset):
    """
    Multimodal dataset that pairs:
      • A triggered JPG image  → visual modality
      • A window of IMU rows   → kinematic modality
      • A severity score       → regression target [1.0, 10.0]

    The IMU window is centred on the trigger event row, with
    `imu_window_size // 2` rows before and after.
    """

    def __init__(
        self,
        df:          pd.DataFrame,
        image_dir:   str,
        imu_extractor: IMUFeatureExtractor,
        transform,
        cfg:         Config,
        is_train:    bool = True,
    ):
        self.df            = df.reset_index(drop=True)
        self.image_dir     = Path(image_dir)
        self.extractor     = imu_extractor
        self.transform     = transform
        self.cfg           = cfg
        self.is_train      = is_train

        # Pre-load full telemetry array for fast window slicing
        self.imu_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
        self.all_imu  = df[self.imu_cols].values.astype(np.float32)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------
    def _get_imu_window(self, idx: int) -> np.ndarray:
        half = self.cfg.imu_window_size // 2
        lo   = max(0, idx - half)
        hi   = min(len(self.all_imu), idx + half)
        window = self.all_imu[lo:hi]
        # Pad to fixed length if at boundaries
        if len(window) < self.cfg.imu_window_size:
            pad_needed = self.cfg.imu_window_size - len(window)
            window = np.vstack([
                window,
                np.zeros((pad_needed, window.shape[1]), dtype=np.float32)
            ])
        return window

    # ------------------------------------------------------------------
    def _resolve_image_path(self, row) -> Optional[Path]:
        """
        Supports images named by timestamp or by integer index.
        Falls back gracefully to a black image if not found.
        """
        candidates = []
        if "image_file" in self.df.columns and pd.notna(row.get("image_file", None)):
            candidates.append(self.image_dir / row["image_file"])
        if "timestamp" in self.df.columns:
            candidates.append(self.image_dir / f"{int(row['timestamp'])}.jpg")
        if "index" in self.df.columns or True:
            candidates.append(self.image_dir / f"{row.name:04d}.jpg")
        for p in candidates:
            if p.exists():
                return p
        return None

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # ── Visual modality ───────────────────────────────────────────
        img_path = self._resolve_image_path(row)
        if img_path is not None:
            img = np.array(Image.open(img_path).convert("RGB"))
        else:
            # Black image fallback for missing files
            img = np.zeros((self.cfg.img_size, self.cfg.img_size, 3), dtype=np.uint8)

        img_tensor = self.transform(image=img)["image"]   # (3, H, W) float32

        # ── IMU modality ──────────────────────────────────────────────
        window      = self._get_imu_window(idx)
        imu_feats   = self.extractor.extract(window)      # (D,) float32
        imu_tensor  = torch.from_numpy(imu_feats)

        if self.is_train:
            imu_tensor = inject_sensor_noise(imu_tensor)

        # ── Target ───────────────────────────────────────────────────
        severity = float(row.get("severity_label", 0.0))
        severity = np.clip(severity, self.cfg.score_min, self.cfg.score_max)
        target   = torch.tensor(severity, dtype=torch.float32)

        return {
            "image":    img_tensor,
            "imu":      imu_tensor,
            "severity": target,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 7.  VISION BACKBONE  (ConvNeXt-Tiny, ImageNet pre-trained)
# ──────────────────────────────────────────────────────────────────────────────
class VisionEncoder(nn.Module):
    """
    ConvNeXt-Tiny backbone with a projection head.

    Why ConvNeXt-Tiny over ViT for ESP32-CAM:
      • Handles low-resolution (224²) images more gracefully than ViT-B/16
      • ~28M params — strong features without over-fitting on small datasets
      • ConvNeXt's depthwise separable design captures local texture well,
        which is critical for crack/pothole texture recognition
      • Alternative: EfficientNetV2-S (swap model name in Config)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.vision_model,
            pretrained=True,
            num_classes=0,     # strip classifier head; return feature map
            global_pool="avg", # global average pool → (B, C)
        )
        backbone_dim = self.backbone.num_features

        if not cfg.freeze_backbone:
            # Unfreeze all — fine-tune end-to-end
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            # Freeze all but last two stages
            for name, p in self.backbone.named_parameters():
                if "stages.3" not in name and "stages.2" not in name:
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
# 10. FULL MULTIMODAL POTHOLE NETWORK
# ──────────────────────────────────────────────────────────────────────────────
class PotholeNet(nn.Module):
    """
    End-to-end multimodal severity regression network.

    Forward pass:
      image (B,3,H,W) ──► VisionEncoder  ──► (B, V)  ─┐
                                                        ├─► CrossAttentionFusion ──► Regressor ──► score (B,)
      imu   (B, D)    ──► IMUEncoder     ──► (B, I)  ─┘

    Output activation: scaled sigmoid → [1.0, 10.0]
      score = 1.0 + 9.0 * sigmoid(logit)
    """

    def __init__(self, imu_feature_dim: int, cfg: Config):
        super().__init__()
        self.vision_enc = VisionEncoder(cfg)
        self.imu_enc    = IMUEncoder(imu_feature_dim, cfg)
        self.fusion     = CrossAttentionFusion(cfg)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(cfg.fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout / 2),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        self.score_min = cfg.score_min
        self.score_max = cfg.score_max
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        image: torch.Tensor,   # (B, 3, H, W)
        imu:   torch.Tensor,   # (B, D)
    ) -> torch.Tensor:
        v     = self.vision_enc(image)            # (B, V)
        i     = self.imu_enc(imu)                 # (B, I)
        fused = self.fusion(v, i)                 # (B, F)
        logit = self.regressor(fused).squeeze(-1) # (B,)

        # Scale sigmoid output to [score_min, score_max]
        score = (
            self.score_min
            + (self.score_max - self.score_min) * torch.sigmoid(logit)
        )
        return score


# ──────────────────────────────────────────────────────────────────────────────
# 11. LOSS FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
class WingLoss(nn.Module):
    """
    Wing Loss (Feng et al., 2018) — originally for facial landmark detection,
    highly effective for ordinal regression with rare extreme values.

    Compared to MSE/MAE:
      • Large gradient for small errors (improves fine-grained accuracy)
      • Logarithmic penalty for large errors (robustness to outliers/rare classes)

    Perfect for pothole scoring: common 3–5 bumps get tight loss;
    rare 8–10 craters don't overwhelm the gradient.
    """

    def __init__(self, w: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.w       = w
        self.epsilon = epsilon
        self.C       = w - w * math.log(1 + w / epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1 + diff / self.epsilon),
            diff - self.C,
        )
        return loss.mean()


class FocalMSELoss(nn.Module):
    """
    Focal MSE: downweights easy (common) examples, upweights hard (rare) ones.
    gamma > 0 increasingly focuses on high-severity rare potholes.
    """

    def __init__(self, gamma: float = 2.0, score_min: float = 1.0, score_max: float = 10.0):
        super().__init__()
        self.gamma     = gamma
        self.score_min = score_min
        self.score_max = score_max

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse    = (pred - target) ** 2
        # Weight by normalized severity — rare high scores get higher weight
        weight = ((target - self.score_min) / (self.score_max - self.score_min)) ** self.gamma
        return (weight * mse).mean()


def build_loss(cfg: Config) -> nn.Module:
    if cfg.loss_type == "wing":
        return WingLoss(w=cfg.wing_w, epsilon=cfg.wing_epsilon)
    elif cfg.loss_type == "huber":
        return nn.HuberLoss(delta=cfg.huber_delta)
    elif cfg.loss_type == "focal_mse":
        return FocalMSELoss(score_min=cfg.score_min, score_max=cfg.score_max)
    else:
        raise ValueError(f"Unknown loss_type: {cfg.loss_type}")


# ──────────────────────────────────────────────────────────────────────────────
# 12. METRICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    mae  = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    # Within-1 accuracy: % predictions within ±1.0 of true score
    w1   = np.mean(np.abs(preds - targets) <= 1.0) * 100.0
    # Rank correlation (Spearman)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(preds, targets)
    return {"MAE": mae, "RMSE": rmse, "Within1%": w1, "SpearmanRho": rho}


# ──────────────────────────────────────────────────────────────────────────────
# 13. WEIGHTED SAMPLER  (handles class imbalance for rare severe potholes)
# ──────────────────────────────────────────────────────────────────────────────
def build_weighted_sampler(labels: np.ndarray, n_bins: int = 9) -> WeightedRandomSampler:
    """
    Discretize continuous scores into bins and over-sample rare high-severity bins.
    """
    bins         = np.linspace(1.0, 10.0, n_bins + 1)
    bin_indices  = np.digitize(labels, bins) - 1
    bin_indices  = np.clip(bin_indices, 0, n_bins - 1)
    bin_counts   = np.bincount(bin_indices, minlength=n_bins).astype(np.float32)
    bin_counts   = np.where(bin_counts == 0, 1.0, bin_counts)
    weights      = 1.0 / bin_counts[bin_indices]
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(weights),
        replacement=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 14. DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_data(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader, IMUFeatureExtractor]:
    df = pd.read_csv(cfg.telemetry_csv)

    # Validate required columns
    required_imu = ["ax", "ay", "az", "gx", "gy", "gz"]
    missing = [c for c in required_imu if c not in df.columns]
    if missing:
        raise ValueError(f"Telemetry CSV missing columns: {missing}")

    has_labels = "severity_label" in df.columns

    # ── Split ─────────────────────────────────────────────────────────
    idx       = np.arange(len(df))
    labels    = df["severity_label"].values if has_labels else np.zeros(len(df))

    idx_tv, idx_test = train_test_split(
        idx, test_size=cfg.test_split, random_state=cfg.seed, shuffle=True
    )
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=cfg.val_split / (1 - cfg.test_split),
        random_state=cfg.seed, shuffle=True
    )

    df_train = df.iloc[idx_train]
    df_val   = df.iloc[idx_val]
    df_test  = df.iloc[idx_test]

    log.info(f"Dataset split — Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # ── Feature extractor ─────────────────────────────────────────────
    extractor = IMUFeatureExtractor(cfg)

    # ── RobustScaler on IMU raw columns (fitted on train only) ────────
    scaler = RobustScaler()
    train_raw_imu_cols = df_train[["ax","ay","az","gx","gy","gz"]].values
    scaler.fit(train_raw_imu_cols)
    df_train = df_train.copy()
    df_val   = df_val.copy()
    df_test  = df_test.copy()
    df_train[["ax","ay","az","gx","gy","gz"]] = scaler.transform(df_train[["ax","ay","az","gx","gy","gz"]].values)
    df_val[["ax","ay","az","gx","gy","gz"]]   = scaler.transform(df_val[["ax","ay","az","gx","gy","gz"]].values)
    df_test[["ax","ay","az","gx","gy","gz"]]  = scaler.transform(df_test[["ax","ay","az","gx","gy","gz"]].values)

    t_transform = build_train_transform(cfg.img_size)
    v_transform = build_val_transform(cfg.img_size)

    ds_train = PotholeDataset(df_train, cfg.image_dir, extractor, t_transform, cfg, is_train=True)
    ds_val   = PotholeDataset(df_val,   cfg.image_dir, extractor, v_transform, cfg, is_train=False)
    ds_test  = PotholeDataset(df_test,  cfg.image_dir, extractor, v_transform, cfg, is_train=False)

    sampler = build_weighted_sampler(labels[idx_train])

    dl_train = DataLoader(
        ds_train, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True,
    )
    dl_val  = DataLoader(
        ds_val, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )
    dl_test = DataLoader(
        ds_test, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )

    return dl_train, dl_val, dl_test, extractor


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
    total_loss, total_mae, n = 0.0, 0.0, 0

    for batch in loader:
        img  = batch["image"].to(device, non_blocking=True)
        imu  = batch["imu"].to(device,   non_blocking=True)
        tgt  = batch["severity"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.use_amp):
            pred = model(img, imu)
            loss = criterion(pred, tgt)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs         = tgt.size(0)
        total_loss += loss.item() * bs
        total_mae  += torch.abs(pred.detach() - tgt).sum().item()
        n          += bs

    return {"loss": total_loss / n, "MAE": total_mae / n}


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
    all_preds, all_tgts = [], []

    for batch in loader:
        img  = batch["image"].to(device, non_blocking=True)
        imu  = batch["imu"].to(device,   non_blocking=True)
        tgt  = batch["severity"].to(device, non_blocking=True)

        with autocast(enabled=cfg.use_amp):
            pred = model(img, imu)
            loss = criterion(pred, tgt)

        bs         = tgt.size(0)
        total_loss += loss.item() * bs
        n          += bs
        all_preds.append(pred.cpu().numpy())
        all_tgts.append(tgt.cpu().numpy())

    preds   = np.concatenate(all_preds)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    dl_train, dl_val, dl_test, extractor = load_data(cfg)

    # ── Model ─────────────────────────────────────────────────────────
    model = PotholeNet(
        imu_feature_dim=extractor.feature_dim,
        cfg=cfg,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────
    criterion = build_loss(cfg)
    # Differential learning rate to preserve pre-trained backbone features
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "vision_enc.backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
            
    optimizer = AdamW([
        {"params": backbone_params, "lr": cfg.lr * 0.1},
        {"params": other_params, "lr": cfg.lr}
    ], weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    
    scheduler    = WarmupCosineScheduler(optimizer, cfg)
    amp_scaler   = GradScaler(enabled=cfg.use_amp)
    early_stop   = EarlyStopping(patience=150)

    writer = SummaryWriter(cfg.log_dir) if HAS_TB else None

    best_val_mae  = float("inf")
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
            f"Train Loss={train_m['loss']:.4f} MAE={train_m['MAE']:.4f} | "
            f"Val   Loss={val_m['loss']:.4f}  MAE={val_m['MAE']:.4f}  "
            f"W1%={val_m['Within1%']:.1f}  ρ={val_m['SpearmanRho']:.3f} | "
            f"LR={lr_now:.2e} | {elapsed:.1f}s"
        )

        # TensorBoard
        if writer:
            writer.add_scalars("Loss",    {"train": train_m["loss"],  "val": val_m["loss"]}, epoch)
            writer.add_scalars("MAE",     {"train": train_m["MAE"],   "val": val_m["MAE"]},  epoch)
            writer.add_scalar("Val/Within1pct",   val_m["Within1%"],       epoch)
            writer.add_scalar("Val/SpearmanRho",  val_m["SpearmanRho"],    epoch)
            writer.add_scalar("LR",               lr_now,                  epoch)

        # Save best checkpoint
        if val_m["MAE"] < best_val_mae:
            best_val_mae = val_m["MAE"]
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_mae":        best_val_mae,
                "cfg":            cfg,
            }, best_ckpt)
            log.info(f"  ✓ New best checkpoint saved (Val MAE={best_val_mae:.4f})")

        if early_stop(val_m["MAE"]):
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
        f"Test MAE={test_m['MAE']:.4f} | RMSE={test_m['RMSE']:.4f} | "
        f"Within-1={test_m['Within1%']:.1f}% | Spearman-ρ={test_m['SpearmanRho']:.3f}"
    )

    return model, test_m


# ──────────────────────────────────────────────────────────────────────────────
# 18. INFERENCE  (production scoring)
# ──────────────────────────────────────────────────────────────────────────────
class PotholeScorer:
    """
    Production-ready inference wrapper.

    Usage:
        scorer = PotholeScorer("checkpoints/best_model.pt")
        score  = scorer.predict("images/00123.jpg", imu_window_np)
        # score ∈ [1.0, 10.0]
    """

    SEVERITY_LABELS = {
        (1.0, 2.5):  ("MINOR",    "Small surface imperfection, no action needed"),
        (2.5, 4.5):  ("MODERATE", "Noticeable bump, monitor for worsening"),
        (4.5, 6.5):  ("SERIOUS",  "Significant pothole, schedule maintenance"),
        (6.5, 8.5):  ("SEVERE",   "Large crater, urgent repair required"),
        (8.5, 10.1): ("CRITICAL", "Dangerous road hazard, immediate closure"),
    }

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg  = ckpt["cfg"]
        self.extractor = IMUFeatureExtractor(cfg)

        self.model = PotholeNet(
            imu_feature_dim=self.extractor.feature_dim,
            cfg=cfg,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.transform = build_val_transform(cfg.img_size)

    @torch.no_grad()
    def predict(
        self,
        image_path:  str,
        imu_window:  np.ndarray,
        return_label: bool = True,
    ) -> Dict:
        # Image
        img    = np.array(Image.open(image_path).convert("RGB"))
        img_t  = self.transform(image=img)["image"].unsqueeze(0).to(self.device)

        # IMU features
        feats  = self.extractor.extract(imu_window)
        imu_t  = torch.from_numpy(feats).unsqueeze(0).to(self.device)

        with autocast(enabled=True):
            score = self.model(img_t, imu_t).item()

        result = {"score": round(score, 2)}

        if return_label:
            for (lo, hi), (label, desc) in self.SEVERITY_LABELS.items():
                if lo <= score < hi:
                    result["severity"]    = label
                    result["description"] = desc
                    break

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        image_paths: List[str],
        imu_windows: List[np.ndarray],
    ) -> List[Dict]:
        return [
            self.predict(p, w)
            for p, w in zip(image_paths, imu_windows)
        ]


# ──────────────────────────────────────────────────────────────────────────────
# 19. SYNTHETIC DATA GENERATOR  (for smoke-testing without real hardware)
# ──────────────────────────────────────────────────────────────────────────────
def generate_synthetic_dataset(
    n_samples:  int   = 500,
    image_dir:  str   = "data/images",
    csv_path:   str   = "data/telemetry.csv",
    seed:       int   = 42,
):
    """
    Creates dummy JPG images (random noise) and telemetry CSV
    so you can run train() end-to-end immediately without real hardware.

    Severity distribution is intentionally imbalanced (right-skewed)
    to mimic real road conditions.
    """
    import os
    os.makedirs(image_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Imbalanced severity: most samples are 2-5, rare are 8-10
    severities = rng.choice(
        np.arange(1.0, 10.1, 0.5),
        size=n_samples,
        p=np.array([
            3,5,8,12,15,15,12,8,6,5,
            4,3,3,2,2,2,1,1,1,1
        ], dtype=float) / 100.0
    ).clip(1.0, 10.0)

    rows = []
    for i, sev in enumerate(severities):
        ts = 1718000000 + i * 100
        # Higher severity → larger Z acceleration spike
        az_spike = 0.5 + sev * 0.8 + rng.normal(0, 0.3)
        rows.append({
            "timestamp":      ts,
            "ax":             rng.normal(0.0, 0.2),
            "ay":             rng.normal(0.0, 0.2),
            "az":             az_spike,
            "gx":             rng.normal(0.0, 0.05),
            "gy":             rng.normal(0.0, 0.05),
            "gz":             rng.normal(0.0, 0.05),
            "severity_label": round(float(sev), 1),
            "image_file":     f"{i:04d}.jpg",
        })

        # Generate dummy image (random noise resembling road)
        img_np = rng.integers(50, 180, size=(96, 96, 3), dtype=np.uint8)
        # Add fake crack lines for high-severity
        if sev > 6:
            for _ in range(int(sev)):
                r = rng.integers(0, 90)
                img_np[r:r+3, :, :] = rng.integers(20, 60, size=(3, 96, 3), dtype=np.uint8)
        Image.fromarray(img_np).save(f"{image_dir}/{i:04d}.jpg")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    log.info(f"Synthetic dataset: {n_samples} samples written to {csv_path}")
    log.info(f"Severity distribution:\n{pd.cut(df['severity_label'], bins=5).value_counts().sort_index()}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 20. ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multimodal Pothole Severity Scorer")
    parser.add_argument("--mode",     type=str, default="train",
                        choices=["train", "infer", "synth"],
                        help="'train': run training | 'infer': single inference | 'synth': generate synthetic data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--image",      type=str, default=None)
    parser.add_argument("--csv_row",    type=int, default=0,
                        help="Row index in telemetry CSV to use as IMU window centre for inference")
    parser.add_argument("--synth_n",   type=int, default=500)
    args = parser.parse_args()

    if args.mode == "synth":
        generate_synthetic_dataset(n_samples=args.synth_n)
        log.info("Synthetic dataset generated. Run with --mode train to start training.")

    elif args.mode == "train":
        # Quick-start: generate synthetic data if no real data present
        if not Path(CFG.telemetry_csv).exists():
            log.warning("No telemetry.csv found. Generating synthetic dataset for demo...")
            generate_synthetic_dataset(n_samples=500)
        train(CFG)

    elif args.mode == "infer":
        if not Path(args.checkpoint).exists():
            log.error(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

        scorer = PotholeScorer(args.checkpoint)
        df_tel = pd.read_csv(CFG.telemetry_csv)

        # Build a window centred on the requested row
        half   = CFG.imu_window_size // 2
        lo     = max(0, args.csv_row - half)
        hi     = min(len(df_tel), args.csv_row + half)
        window = df_tel.iloc[lo:hi][["ax","ay","az","gx","gy","gz"]].values.astype(np.float32)

        img_path = args.image or (Path(CFG.image_dir) / f"{args.csv_row:04d}.jpg")
        result   = scorer.predict(str(img_path), window)

        print("\n" + "="*50)
        print(f"  POTHOLE SEVERITY SCORE:  {result['score']:.2f} / 10.0")
        print(f"  CLASS:                   {result.get('severity', 'N/A')}")
        print(f"  DESCRIPTION:             {result.get('description', '')}")
        print("="*50 + "\n")
