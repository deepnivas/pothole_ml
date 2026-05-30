"""Upload-friendly pothole prediction entrypoint.

This script wraps the DeepLabv3+ segmentation service and provides:
- CLI prediction from a single image path
- A small Gradio app for image upload and live output

It now supports two backends:
 - "segmentation": the DeepLabV3+ service (mask + overlay)
 - "cnn": image-only CNN classifier (PotholeScorer) which returns probability

Usage examples:
  python3 pothole_ml/predict_seg.py --image path/to/image.jpg --model cnn
  python3 pothole_ml/predict_seg.py --serve
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from pothole_deeplabv3plus import PotholeSegmentationService
try:
    from pothole_multimodal_system import PotholeScorer
except Exception:
    PotholeScorer = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

SEGMENTATION_DEFAULTS = [
    SCRIPT_DIR / "checkpoints" / "pothole_deeplabv3plus.pt",
    REPO_ROOT / "checkpoints" / "pothole_deeplabv3plus.pt",
]

CNN_DEFAULTS = [
    SCRIPT_DIR / "checkpoints" / "best_model.pt",
    REPO_ROOT / "checkpoints" / "best_model.pt",
]


def _resolve_checkpoint(checkpoint_path: str | None, defaults: list[Path]) -> Path:
    if checkpoint_path:
        checkpoint = Path(checkpoint_path)
        if checkpoint.exists():
            return checkpoint
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    for candidate in defaults:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Checkpoint not found. Looked in: " + ", ".join(str(path) for path in defaults)
    )


def _load_segmentation_service(checkpoint_path: str | None) -> PotholeSegmentationService:
    checkpoint = _resolve_checkpoint(checkpoint_path, SEGMENTATION_DEFAULTS)
    return PotholeSegmentationService(checkpoint_path=str(checkpoint))


def _load_cnn_scorer(checkpoint_path: str | None) -> "PotholeScorer":
    if PotholeScorer is None:
        raise ImportError("PotholeScorer (CNN) not available. Ensure pothole_multimodal_system.py is importable.")
    checkpoint = _resolve_checkpoint(checkpoint_path, CNN_DEFAULTS)
    return PotholeScorer(str(checkpoint))


def _format_summary(details: dict) -> str:
    return (
        f"Severity: {details['severity_label']}\n"
        f"Score: {details['severity_score']:.2f} / 10.0\n"
        f"Confidence: {details['confidence']:.4f}\n"
        f"Mask area ratio: {details['mask_area_ratio']:.6f}\n"
        f"Has pothole: {'yes' if details['has_pothole'] else 'no'}"
    )


def _build_cnn_overlay(image_path: str, result: dict) -> np.ndarray:
    # Create a simple overlay by drawing the predicted label and probability onto the image
    from PIL import ImageDraw, ImageFont

    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text = f"{result['label']} ({result['probability']:.3f})"
    # Draw a translucent bar and text
    draw.rectangle([(0, 0), (im.width, 28)], fill=(0, 0, 0, 200))
    draw.text((6, 6), text, fill=(255, 255, 255), font=font)
    return np.array(im)


def predict_image(image_path: str, model_type: str = "segmentation", checkpoint_path: str | None = None) -> Tuple[str, np.ndarray]:
    if model_type == "cnn":
        scorer = _load_cnn_scorer(checkpoint_path)
        result = scorer.predict(image_path)
        summary = (
            f"Class: {result['label']}\nProbability: {result['probability']:.4f}\nThreshold: {result['threshold']:.4f}"
        )
        overlay = _build_cnn_overlay(image_path, result)
        return summary, overlay

    service = _load_segmentation_service(checkpoint_path)
    details = service.predict_with_details(image_path)
    overlay = service.build_overlay(details["original_image"], details["mask"])
    return _format_summary(details), overlay


def predict_pil_image(image: Image.Image, model_type: str = "segmentation", checkpoint_path: str | None = None) -> Tuple[str, np.ndarray]:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        image.convert("RGB").save(temp_path)
        return predict_image(str(temp_path), model_type=model_type, checkpoint_path=checkpoint_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def build_app(checkpoint_path: str | None = None):
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "Gradio is not installed. Install it with `pip install gradio` to use the upload UI."
        ) from exc

    # Pre-load segmentation service is not necessary; we'll load per-request depending on model selector

    def _predict(uploaded_image_path, model_selector, ckpt_override):
        if uploaded_image_path is None:
            return "Upload an image to get a prediction.", None

        image = Image.open(uploaded_image_path).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            image.save(temp_path)
            if model_selector == "cnn":
                summary, overlay = predict_image(str(temp_path), model_type="cnn", checkpoint_path=ckpt_override or checkpoint_path)
            else:
                summary, overlay = predict_image(str(temp_path), model_type="segmentation", checkpoint_path=ckpt_override or checkpoint_path)
            return summary, overlay
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    with gr.Blocks(title="Pothole Prediction") as demo:
        gr.Markdown("# Pothole Prediction")
        gr.Markdown("Upload a road image to get pothole severity and mask overlay.")
        with gr.Row():
            image_input = gr.Image(type="filepath", label="Upload image")
            model_selector = gr.Radio(choices=["segmentation", "cnn"], value="segmentation", label="Model")
            ckpt_input = gr.Textbox(label="Checkpoint path (optional)", placeholder="Leave blank to use defaults")
            image_output = gr.Image(type="numpy", label="Prediction overlay")
        text_output = gr.Textbox(label="Prediction details", lines=5)
        run_button = gr.Button("Predict")

        run_button.click(fn=_predict, inputs=[image_input, model_selector, ckpt_input], outputs=[text_output, image_output])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload-friendly pothole predictor")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image", type=str, default=None, help="Single image path for CLI prediction")
    parser.add_argument("--serve", action="store_true", help="Launch a Gradio upload UI")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    parser.add_argument("--model", type=str, default="segmentation", choices=["segmentation", "cnn"], help="Which model/backend to use for CLI prediction or default UI selection")
    args = parser.parse_args()

    if args.serve:
        app = build_app(checkpoint_path=args.checkpoint)
        app.launch(share=args.share)
        return

    if args.image is None:
        parser.error("Provide --image for CLI prediction or use --serve for the upload UI")

    summary, overlay = predict_image(args.image, model_type=args.model, checkpoint_path=args.checkpoint)
    print(summary)
    output_path = Path(args.image).with_name(f"{Path(args.image).stem}_overlay.png")
    Image.fromarray(overlay).save(output_path)
    print(f"Overlay saved to: {output_path}")


if __name__ == "__main__":
    main()
