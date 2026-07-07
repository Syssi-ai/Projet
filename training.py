import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ---------------- Config ----------------
DATA_DIR = "archive/imgs_zip/imgs" #DIR: dossier dans lequel il y a la data set
BATCH_SIZE = 16 #nombre d'images dans chaque lot
NUM_EPOCHS = 10  #nombre de fois que je vais faire le passage complet de la donnée
VAL_SPLIT = 0.2 #proportion de données à utiliser pour la validation
SEED = 42  #variable aléatoire
DEVICE = torch.device("cpu")  #device sur lequel les calculs seront effectués

IMAGENET_MEAN = [0.485, 0.456, 0.406] #moyenne des canaux de couleur pour la normalisation
IMAGENET_STD  = [0.229, 0.224, 0.225] #l'écart type des canaux de couleur pour la normalisation

# ---------------- Transforms ----------------
train_tfms = transforms.Compose([ #déclare une variable train_tfms qui contient une série de transformations à appliquer aux images d'entraînement
    transforms.RandomResizedCrop(160, scale=(0.8, 1.0)), #redimensionne aléatoirement l'image à une taille de 160x160 pixels, en conservant une partie de l'image comprise entre 80% et 100% de sa taille originale
    transforms.RandomHorizontalFlip(), #retourner l'image horizontalement avec une probabilité de 50%
    transforms.ColorJitter(0.2, 0.2, 0.2), #1er = lumonosité, 2ème = contraste, 3ème = saturation
    transforms.ToTensor(),#foncion prédéfinie, convertit l'objet en type image 
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), #formule = (image - mean) / std, normalise les valeurs des pixels de l'image en soustrayant la moyenne et en divisant par l'écart type pour chaque canal de couleur
])

val_tfms = transforms.Compose([
    transforms.Resize(180),
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------- Dataset ----------------
# Load once to get class info + consistent split indices
base_dataset = datasets.ImageFolder(DATA_DIR)
class_names = base_dataset.classes
num_classes = len(class_names)
print(f"Found {num_classes} classes, {len(base_dataset)} images")

n_val = int(VAL_SPLIT * len(base_dataset))
n_train = len(base_dataset) - n_val

generator = torch.Generator().manual_seed(SEED) #garantit la séparation des données et l'ordre aléatoire de celles-ci
train_subset, val_subset = random_split(base_dataset, [n_train, n_val], generator=generator)#crée deux sous-ensembles de données à partir du jeu de données de base, un pour l'entraînement et un pour la validation, en utilisant les indices générés par le générateur aléatoire
train_indices = train_subset.indices
val_indices = val_subset.indices

# Two separate ImageFolder instances with different transforms,
# sharing the same underlying file order (so indices line up)
train_full = datasets.ImageFolder(DATA_DIR, transform=train_tfms)#sauvegarde des images
val_full = datasets.ImageFolder(DATA_DIR, transform=val_tfms)#sauvegarde des images

train_ds = torch.utils.data.Subset(train_full, train_indices)  #data set
val_ds = torch.utils.data.Subset(val_full, val_indices)   #data set

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2) #shuffle = mélange, veut qu'il y ai le mélange
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2) #pas la peine à la validation 