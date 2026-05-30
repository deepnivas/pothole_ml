"""DeepLabv3+ pothole segmentation and severity scoring.

This module is designed for the backend side of the pothole workflow:
- Train a segmentation model that predicts pothole masks.
- Convert the predicted mask area into a severity score.
- Support weak supervision from Pascal VOC bounding boxes when true masks
  are not available yet.

Expected dataset layouts:
- images_dir: JPG/PNG road images
- masks_dir: optional binary mask images with the same base filename as images
- annotations_dir: optional Pascal VOC XML files used to generate weak masks

If masks_dir is missing, the dataset can synthesize rectangular weak masks from
VOC bounding boxes so you can bootstrap training from the current repository
assets.
"""

from __future__ import annotations

import math
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class SegmentationConfig:
    image_size: int = 256
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 1e-4
    num_workers: int = 0
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    mask_threshold: float = 0.65
    postprocess_quantile: float = 0.92
    postprocess_min_area_ratio: float = 0.003
    postprocess_max_area_ratio: float = 0.25
    severity_min: float = 1.0
    severity_max: float = 10.0
    max_severity_area_ratio: float = 0.25
    weak_mask_min_area: int = 80
    weak_mask_roi_start_ratio: float = 0.25
    seed: int = 42


DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent / "pothole_image_data" / "Pothole_Image_Data"


def get_best_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_supports_amp(device: torch.device) -> bool:
    return device.type == "cuda"


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(p=0.25),
            A.MotionBlur(blur_limit=5, p=0.15),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_eval_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class PotholeSegmentationDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        masks_dir: Optional[Path] = None,
        annotations_dir: Optional[Path] = None,
        transform: Optional[A.Compose] = None,
        include_synthetic_negatives: bool = False,
    ):
        self.samples: List[Tuple[Path, bool]] = []
        for image_path in image_paths:
            self.samples.append((image_path, False))
            if include_synthetic_negatives:
                self.samples.append((image_path, True))
        self.masks_dir = masks_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self._weak_mask_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_path: Path) -> np.ndarray:
        return np.array(Image.open(image_path).convert("RGB"))

    def _load_mask_from_png(self, image_path: Path) -> Optional[np.ndarray]:
        if self.masks_dir is None:
            return None
        candidates = [
            self.masks_dir / f"{image_path.stem}.png",
            self.masks_dir / f"{image_path.stem}.jpg",
            self.masks_dir / f"{image_path.stem}.jpeg",
        ]
        for candidate in candidates:
            if candidate.exists():
                mask = np.array(Image.open(candidate).convert("L"))
                return (mask > 0).astype(np.float32)
        return None

    def _load_mask_from_voc(self, image_path: Path, image_shape: Tuple[int, int]) -> np.ndarray:
        mask = np.zeros(image_shape, dtype=np.float32)
        if self.annotations_dir is None:
            return mask

        xml_path = self.annotations_dir / f"{image_path.stem}.xml"
        if not xml_path.exists():
            return mask

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            return mask

        for obj in root.findall("object"):
            name = obj.findtext("name", default="")
            if name.lower() != "pothole":
                continue
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin = int(float(bndbox.findtext("xmin", default="0")))
            ymin = int(float(bndbox.findtext("ymin", default="0")))
            xmax = int(float(bndbox.findtext("xmax", default="0")))
            ymax = int(float(bndbox.findtext("ymax", default="0")))
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(image_shape[1], xmax)
            ymax = min(image_shape[0], ymax)
            if xmax > xmin and ymax > ymin:
                mask[ymin:ymax, xmin:xmax] = 1.0
        return mask

    def _generate_weak_mask(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        roi_start = int(h * 0.25)
        roi = gray[roi_start:, :]

        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        thresh_val, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        weak_mask = np.zeros((h, w), dtype=np.float32)
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area < 80:
                continue
            x = stats[label_idx, cv2.CC_STAT_LEFT]
            y = stats[label_idx, cv2.CC_STAT_TOP]
            width = stats[label_idx, cv2.CC_STAT_WIDTH]
            height = stats[label_idx, cv2.CC_STAT_HEIGHT]
            weak_mask[roi_start + y : roi_start + y + height, x : x + width] = (labels[y : y + height, x : x + width] == label_idx).astype(np.float32)

        if weak_mask.sum() == 0:
            dark_threshold = np.percentile(gray, 25)
            weak_mask = (gray < dark_threshold).astype(np.float32)
            weak_mask[:roi_start, :] = 0.0

        return weak_mask

    def _weak_mask_cache_key(self, image_path: Path) -> str:
        return str(image_path.resolve())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path, is_synthetic_negative = self.samples[idx]
        image = self._load_image(image_path)
        if is_synthetic_negative:
            crop_height = max(1, int(image.shape[0] * 0.45))
            image = image[:crop_height, :, :]
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        else:
            mask = self._load_mask_from_png(image_path)
            if mask is None:
                mask = self._load_mask_from_voc(image_path, image.shape[:2])
            if mask.sum() == 0:
                cache_key = self._weak_mask_cache_key(image_path)
                cached_mask = self._weak_mask_cache.get(cache_key)
                if cached_mask is None:
                    cached_mask = self._generate_weak_mask(image)
                    self._weak_mask_cache[cache_key] = cached_mask
                mask = cached_mask

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask)

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.float().unsqueeze(0)

        return {"image": image, "mask": mask, "image_path": str(image_path)}


class DeepLabv3PlusPotholeModel(nn.Module):
    def __init__(self, encoder_name: str = "resnet34", encoder_weights: str = "imagenet"):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SeverityScorer:
    def __init__(
        self,
        severity_min: float = 1.0,
        severity_max: float = 10.0,
        max_severity_area_ratio: float = 0.25,
    ):
        self.severity_min = severity_min
        self.severity_max = severity_max
        self.max_severity_area_ratio = max_severity_area_ratio

    def score(self, mask: np.ndarray, confidence: float = 1.0) -> Tuple[float, str]:
        if mask.ndim == 3:
            mask = mask.squeeze()
        area_ratio = float(mask.sum()) / float(mask.size + 1e-8)
        normalized = np.clip(area_ratio / self.max_severity_area_ratio, 0.0, 1.0)
        score = self.severity_min + (self.severity_max - self.severity_min) * math.sqrt(normalized)
        score *= float(np.clip(confidence, 0.5, 1.0))
        score = float(np.clip(score, self.severity_min, self.severity_max))
        label = self.to_label(score)
        return round(score, 2), label

    @staticmethod
    def to_label(score: float) -> str:
        if score < 2.5:
            return "MINOR"
        if score < 4.5:
            return "MODERATE"
        if score < 6.5:
            return "SEVERE"
        return "CRITICAL"


class PotholeSegmentationService:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[SegmentationConfig] = None,
    ):
        self.cfg = config or SegmentationConfig()
        seed_everything(self.cfg.seed)
        self.device = torch.device(device) if device is not None else get_best_device()
        self.use_amp = device_supports_amp(self.device)
        log.info(
            "Apple MPS detected: %s | Using device: %s | AMP: %s",
            "yes" if self.device.type == "mps" else "no",
            self.device,
            "enabled" if self.use_amp else "disabled",
        )
        self.model = DeepLabv3PlusPotholeModel(
            encoder_name=self.cfg.encoder_name,
            encoder_weights=None,
        ).to(self.device)
        self.transform = build_eval_transform(self.cfg.image_size)
        self.scorer = SeverityScorer(
            severity_min=self.cfg.severity_min,
            severity_max=self.cfg.severity_max,
            max_severity_area_ratio=self.cfg.max_severity_area_ratio,
        )

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=self.device)
            model_state = state.get("model_state", state)
            self.model.load_state_dict(model_state)
        self.model.eval()

    @torch.no_grad()
    def predict_with_details(self, image_path: str) -> Dict[str, object]:
        image = np.array(Image.open(image_path).convert("RGB"))
        image_t = self.transform(image=image)["image"].unsqueeze(0).to(self.device)
        logits = self.model(image_t)
        probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
        mask = self._postprocess_probability_map(probs)
        confidence = float(probs[mask > 0].mean()) if mask.any() else float(probs.max())
        score, label = self.scorer.score(mask, confidence=confidence)
        return {
            "severity_score": score,
            "severity_label": label,
            "mask_area_ratio": round(float(mask.sum()) / float(mask.size + 1e-8), 6),
            "confidence": round(confidence, 4),
            "has_pothole": bool(mask.any()),
            "mask": mask,
            "probability_map": probs,
            "original_image": image,
        }

    def _postprocess_probability_map(self, probs: np.ndarray) -> np.ndarray:
        smoothed = cv2.GaussianBlur(probs.astype(np.float32), (5, 5), 0)
        threshold = max(self.cfg.mask_threshold, float(np.quantile(smoothed, self.cfg.postprocess_quantile)))
        mask = (smoothed >= threshold).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        min_area = max(8, int(mask.size * self.cfg.postprocess_min_area_ratio))
        max_area = int(mask.size * self.cfg.postprocess_max_area_ratio)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)

        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                filtered[labels == label_idx] = 1

        if filtered.sum() == 0 and num_labels > 1:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[largest_label, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == largest_label] = 1

        if filtered.sum() == 0:
            peak_y, peak_x = np.unravel_index(int(np.argmax(smoothed)), smoothed.shape)
            radius = max(3, int(min(smoothed.shape[:2]) * 0.05))
            cv2.circle(filtered, (int(peak_x), int(peak_y)), radius, 1, -1)

        return filtered.astype(np.uint8)

    @staticmethod
    def build_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        if mask.ndim == 3:
            mask = mask.squeeze()
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        overlay = image.copy()
        red = np.zeros_like(overlay)
        red[..., 0] = 255
        mask_3d = np.repeat(mask[:, :, None].astype(bool), 3, axis=2)
        overlay[mask_3d] = (overlay[mask_3d] * (1.0 - alpha) + red[mask_3d] * alpha).astype(np.uint8)
        return overlay

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict[str, object]:
        details = self.predict_with_details(image_path)
        details.pop("mask", None)
        details.pop("probability_map", None)
        details.pop("original_image", None)
        return details


def build_dataloaders(
    images_dir: str = str(DEFAULT_IMAGE_DIR),
    masks_dir: Optional[str] = None,
    annotations_dir: Optional[str] = None,
    image_size: int = 256,
    batch_size: int = 8,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    log.info("Scanning images in %s", images_dir)
    if masks_dir is None and annotations_dir is None:
        log.warning(
            "No masks_dir or annotations_dir provided. Training will rely on weak pseudo-labels, so accuracy may stay limited."
        )
    image_dir_path = Path(images_dir)
    image_paths = sorted(
        [
            *image_dir_path.glob("*.jpg"),
            *image_dir_path.glob("*.jpeg"),
            *image_dir_path.glob("*.png"),
        ]
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    log.info("Found %d images", len(image_paths))

    train_paths, val_paths = train_test_split(image_paths, test_size=val_split, random_state=seed, shuffle=True)
    log.info("Split dataset into %d train / %d val", len(train_paths), len(val_paths))
    train_ds = PotholeSegmentationDataset(
        train_paths,
        masks_dir=Path(masks_dir) if masks_dir else None,
        annotations_dir=Path(annotations_dir) if annotations_dir else None,
        transform=build_train_transform(image_size),
        include_synthetic_negatives=True,
    )
    val_ds = PotholeSegmentationDataset(
        val_paths,
        masks_dir=Path(masks_dir) if masks_dir else None,
        annotations_dir=Path(annotations_dir) if annotations_dir else None,
        transform=build_eval_transform(image_size),
        include_synthetic_negatives=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()

    preds_f = preds.view(preds.size(0), -1)
    targets_f = targets.view(targets.size(0), -1)

    tp = (preds_f * targets_f).sum(dim=1)
    fp = (preds_f * (1.0 - targets_f)).sum(dim=1)
    fn = ((1.0 - preds_f) * targets_f).sum(dim=1)
    tn = ((1.0 - preds_f) * (1.0 - targets_f)).sum(dim=1)

    pixel_acc = ((tp + tn) / (tp + tn + fp + fn + 1e-8)).mean().item()
    dice = ((2.0 * tp + 1e-8) / (2.0 * tp + fp + fn + 1e-8)).mean().item()
    iou = ((tp + 1e-8) / (tp + fp + fn + 1e-8)).mean().item()

    return {
        "pixel_acc": pixel_acc,
        "dice": dice,
        "iou": iou,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_pixel_acc = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batch_count = 0
    criterion = nn.BCEWithLogitsLoss()
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = criterion(logits, mask) + dice_loss(logits, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * image.size(0)
        metrics = segmentation_metrics(logits.detach(), mask)
        total_pixel_acc += metrics["pixel_acc"]
        total_dice += metrics["dice"]
        total_iou += metrics["iou"]
        batch_count += 1
        if batch_idx == 1 or batch_idx == total_batches or batch_idx % 10 == 0:
            log.info(
                "Train batch %d/%d | loss=%.4f | pix_acc=%.4f | dice=%.4f | iou=%.4f",
                batch_idx,
                total_batches,
                loss.item(),
                metrics["pixel_acc"],
                metrics["dice"],
                metrics["iou"],
            )
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "pixel_acc": total_pixel_acc / max(batch_count, 1),
        "dice": total_dice / max(batch_count, 1),
        "iou": total_iou / max(batch_count, 1),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_pixel_acc = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batch_count = 0
    criterion = nn.BCEWithLogitsLoss()
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        logits = model(image)
        loss = criterion(logits, mask) + dice_loss(logits, mask)
        total_loss += loss.item() * image.size(0)
        metrics = segmentation_metrics(logits.detach(), mask)
        total_pixel_acc += metrics["pixel_acc"]
        total_dice += metrics["dice"]
        total_iou += metrics["iou"]
        batch_count += 1
        if batch_idx == 1 or batch_idx == total_batches or batch_idx % 10 == 0:
            log.info(
                "Val batch %d/%d | loss=%.4f | pix_acc=%.4f | dice=%.4f | iou=%.4f",
                batch_idx,
                total_batches,
                loss.item(),
                metrics["pixel_acc"],
                metrics["dice"],
                metrics["iou"],
            )
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "pixel_acc": total_pixel_acc / max(batch_count, 1),
        "dice": total_dice / max(batch_count, 1),
        "iou": total_iou / max(batch_count, 1),
    }


def train_model(
    images_dir: str,
    checkpoint_path: str,
    masks_dir: Optional[str] = None,
    annotations_dir: Optional[str] = None,
    config: Optional[SegmentationConfig] = None,
) -> Dict[str, float]:
    cfg = config or SegmentationConfig()
    seed_everything(cfg.seed)
    device = get_best_device()
    use_amp = device_supports_amp(device)
    log.info(
        "Apple MPS detected: %s | Using device: %s | AMP: %s",
        "yes" if device.type == "mps" else "no",
        device,
        "enabled" if use_amp else "disabled",
    )
    train_loader, val_loader = build_dataloaders(
        images_dir=images_dir,
        masks_dir=masks_dir,
        annotations_dir=annotations_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        val_split=0.2,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
    )

    model = DeepLabv3PlusPotholeModel(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    log.info("Starting training for %d epochs", cfg.epochs)

    best_val = float("inf")
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    history: Dict[str, float] = {}
    for epoch in range(1, cfg.epochs + 1):
        log.info("Epoch %d/%d started", epoch, cfg.epochs)
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        history = {
            "train_loss": train_loss["loss"],
            "train_pixel_acc": train_loss["pixel_acc"],
            "train_dice": train_loss["dice"],
            "train_iou": train_loss["iou"],
            "val_loss": val_loss["loss"],
            "val_pixel_acc": val_loss["pixel_acc"],
            "val_dice": val_loss["dice"],
            "val_iou": val_loss["iou"],
        }
        if val_loss["loss"] < best_val:
            best_val = val_loss["loss"]
            torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, checkpoint)
            log.info("New best checkpoint saved to %s", checkpoint)
        log.info(
            "Epoch %d/%d complete | train_loss=%.4f | train_pix_acc=%.4f | train_dice=%.4f | train_iou=%.4f | val_loss=%.4f | val_pix_acc=%.4f | val_dice=%.4f | val_iou=%.4f",
            epoch,
            cfg.epochs,
            train_loss["loss"],
            train_loss["pixel_acc"],
            train_loss["dice"],
            train_loss["iou"],
            val_loss["loss"],
            val_loss["pixel_acc"],
            val_loss["dice"],
            val_loss["iou"],
        )
    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepLabv3+ pothole detector and severity scorer")
    parser.add_argument("--images_dir", type=str, default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pothole_deeplabv3plus.pt")
    parser.add_argument("--masks_dir", type=str, default=None)
    parser.add_argument("--annotations_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["train", "infer"], default="train")
    parser.add_argument("--image", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        metrics = train_model(
            images_dir=args.images_dir,
            checkpoint_path=args.checkpoint,
            masks_dir=args.masks_dir,
            annotations_dir=args.annotations_dir,
        )
        print(metrics)
    else:
        if args.image is None:
            raise SystemExit("--image is required in infer mode")
        service = PotholeSegmentationService(checkpoint_path=args.checkpoint)
        print(service.predict(args.image))
