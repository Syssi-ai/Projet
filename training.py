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