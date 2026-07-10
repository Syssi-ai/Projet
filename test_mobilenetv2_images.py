import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.models import load_model


def load_and_prepare_image(image_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    """Load an image from disk and prepare a batch for MobileNetV2-style inputs."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    return preprocess_input(array)


def topk_custom(preds: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return top-k (class_index, probability) for non-ImageNet output heads."""
    probs = preds[0]
    k = min(k, probs.shape[0])
    idx = np.argsort(probs)[::-1][:k]
    return [(int(i), float(probs[i])) for i in idx]


def gather_images(single_image: str | None, folder: str | None) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths: list[Path] = []

    if single_image:
        p = Path(single_image)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        paths.append(p)

    if folder:
        d = Path(folder)
        if not d.is_dir():
            raise NotADirectoryError(f"Folder not found: {d}")
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in exts:
                paths.append(p)

    if not paths:
        raise ValueError("No images found. Use --image or --folder.")

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick tester for mobilenetv2.keras on images")
    parser.add_argument("--model", default="mobilenetv2.keras", help="Path to .keras model")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--folder", help="Folder of images (searched recursively)")
    parser.add_argument("--top-k", type=int, default=3, help="How many predictions to show")
    args = parser.parse_args()

    model = load_model(args.model)

    # Expected shape is usually (None, 224, 224, 3) for MobileNetV2.
    _, h, w, _ = model.input_shape
    target_size = (w, h)

    image_paths = gather_images(args.image, args.folder)

    for image_path in image_paths:
        batch = load_and_prepare_image(image_path, target_size)
        preds = model.predict(batch, verbose=0)

        print(f"\n{image_path}")
        if preds.shape[-1] == 1000:
            decoded = decode_predictions(preds, top=args.top_k)[0]
            for rank, (_, label, prob) in enumerate(decoded, start=1):
                print(f"  {rank}. {label:<20} {prob * 100:.2f}%")
        else:
            for rank, (class_idx, prob) in enumerate(topk_custom(preds, args.top_k), start=1):
                print(f"  {rank}. class_{class_idx:<6} {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
