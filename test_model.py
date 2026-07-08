import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 160
DEFAULT_MODEL_PATH = "best_car_brand_model.pth"

predict_tfms = transforms.Compose([
    transforms.Resize(180),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names = checkpoint.get("class_names")
    if not class_names:
        raise ValueError("The checkpoint does not contain class_names.")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError(
            "The checkpoint does not contain model_state_dict. "
            "Re-train using train.py or save the model weights in the checkpoint."
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_names


def predict_image(model, class_names, image_path: str, device: torch.device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = predict_tfms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, index = torch.max(probabilities, dim=0)

    return class_names[index.item()], confidence.item()


def main():
    parser = argparse.ArgumentParser(
        description="Predict the car brand for a single image using best_car_brand_model.pth"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained checkpoint file.",
    )
    parser.add_argument(
        "--image",
        help="Path to the image to classify. If omitted, you will be prompted.",
    )
    args = parser.parse_args()

    image_path = args.image or input("Enter the image path: ").strip()
    if not image_path:
        raise SystemExit("No image path provided.")

    if not Path(image_path).exists():
        raise SystemExit(f"Image not found: {image_path}")

    if not Path(args.model).exists():
        raise SystemExit(f"Model file not found: {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(args.model, device)
    prediction, confidence = predict_image(model, class_names, image_path, device)

    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
