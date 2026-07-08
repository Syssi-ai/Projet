"""
Script de test pour un modèle PyTorch (.pth) de classification de marques de voitures.

Utilisation :
    python test_model.py --model chemin/vers/model.pth --image chemin/vers/image.jpg
    python test_model.py --model model.pth --image_dir dossier_images/

Avant de lancer :
    1. Renseigne la liste CLASS_NAMES avec les classes dans le MÊME ORDRE
       que celui utilisé pendant l'entraînement (souvent l'ordre alphabétique
       des dossiers si tu as utilisé ImageFolder).
    2. Choisis la bonne architecture (ARCHITECTURE) et adapte-la si besoin.
    3. Vérifie la taille d'image (IMG_SIZE) utilisée à l'entraînement.
"""

import argparse
import os
import json

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------------------------------------------------------------------
# CONFIGURATION À ADAPTER SELON TON ENTRAÎNEMENT
# ---------------------------------------------------------------------------

# Liste des classes dans le même ordre que pendant l'entraînement.
# Exemple : ["Audi", "BMW", "Citroen", "Mercedes", "Peugeot", "Renault", "Toyota"]
CLASS_NAMES = [
    "Audi", "BMW", "Citroen", "Mercedes", "Peugeot", "Renault", "Toyota"
]

# Architecture utilisée pendant l'entraînement.
# Options courantes : "resnet18", "resnet34", "resnet50", "efficientnet_b0", "mobilenet_v2"
ARCHITECTURE = "resnet18"

# Taille d'image attendue par le modèle (souvent 224x224)
IMG_SIZE = 224

# Normalisation standard ImageNet (à garder si tu as utilisé un modèle pré-entraîné)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------


def build_model(architecture, num_classes):
    """Construit l'architecture et adapte la dernière couche au nombre de classes."""
    architecture = architecture.lower()

    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Architecture non gérée : {architecture}")

    return model


def load_model(model_path, architecture, num_classes, device):
    model = build_model(architecture, num_classes)

    checkpoint = torch.load(model_path, map_location=device)

    # Gère le cas où le .pth contient directement un state_dict,
    # ou un dict avec une clé "state_dict" / "model_state_dict"
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def predict_image(model, image_path, transform, device, class_names, topk=3):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top_probs, top_idxs = torch.topk(probabilities, min(topk, len(class_names)))

    results = [
        (class_names[idx], float(prob))
        for idx, prob in zip(top_idxs.tolist(), top_probs.tolist())
    ]
    return results


def main():
    parser = argparse.ArgumentParser(description="Teste un modèle de classification de marques de voitures.")
    parser.add_argument("--model", required=True, help="Chemin vers le fichier .pth")
    parser.add_argument("--image", help="Chemin vers une image unique à tester")
    parser.add_argument("--image_dir", help="Dossier contenant plusieurs images à tester")
    parser.add_argument("--architecture", default=ARCHITECTURE, help="Architecture du modèle")
    parser.add_argument("--topk", type=int, default=3, help="Nombre de prédictions à afficher")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Précise --image ou --image_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    model = load_model(args.model, args.architecture, len(CLASS_NAMES), device)
    transform = get_transform()

    # Construit la liste des images à tester
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for fname in sorted(os.listdir(args.image_dir)):
            if fname.lower().endswith(valid_ext):
                image_paths.append(os.path.join(args.image_dir, fname))

    all_results = {}
    for img_path in image_paths:
        try:
            results = predict_image(model, img_path, transform, device, CLASS_NAMES, args.topk)
            all_results[img_path] = results

            print(f"\nImage : {img_path}")
            for rank, (label, prob) in enumerate(results, start=1):
                print(f"  {rank}. {label} — {prob * 100:.2f}%")
        except Exception as e:
            print(f"Erreur avec {img_path} : {e}")

    # Sauvegarde des résultats en JSON
    output_json = "resultats_predictions.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nRésultats sauvegardés dans {output_json}")


if __name__ == "__main__":
    main()
