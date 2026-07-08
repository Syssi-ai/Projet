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

# ---------------- Model ----------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) #resnet = fonction préséfinie, modèle de réseau de neurones convolutifs pour la classification d'images, pré-entraîné sur le jeu de données ImageNet

for param in model.parameters(): #sauvegarde des paramètres du modèle, itère sur tous les paramètres du modèle et les rend non entraînables, ce qui signifie que leurs valeurs ne seront pas mises à jour pendant l'entraînement
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)#fc = la dernière couche, reprend notre résultat et l'intègre dans le modèle prédéfini
model = model.to(DEVICE)#la variable DEVICE est utilisée pour spécifier l'appareil sur lequel le modèle sera exécuté, dans ce cas, le CPU. Le modèle est déplacé vers cet appareil à l'aide de la méthode .to().

criterion = nn.CrossEntropyLoss()#fonction qui calcule l'erreur du modèle
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3) #adam = optimisuer, ce sont des variables qu'on mofifie pour réduire la perte

# ---------------- Train / Eval loops ----------------
def train_one_epoch(loader):
    model.train() #entraînement du modèle
    total_loss, correct, total = 0.0, 0, 0 
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE) #x= image, y=label
        optimizer.zero_grad() # nouveau calcul entre chaque lot, réinitialise les gradients des paramètres du modèle à zéro avant de calculer les gradients pour le lot actuel
        out = model(x) # le modèle prédit les classes des images d'entrée x, et les résultats sont stockés dans la variable out
        loss = criterion(out, y) # la perte est calculée en comparant les prédictions du modèle (out) avec les étiquettes réelles (y) à l'aide de la fonction de perte définie précédemment (criterion)
        loss.backward() 
        optimizer.step() #améliore les paramètres du modèle en fonction des gradients calculés lors de la rétropropagation
        total_loss += loss.item() * x.size(0) 
        correct += (out.argmax(1) == y).sum().item() 
        total += x.size(0)
    return total_loss / total, correct / total #attention, c'est des divisions, 1ère=perte moyenne, 2ème=exactitude moyenne

@torch.no_grad() #décorateur qui commence par @, désactiver le calcul donc arreter l'entrainement
def evaluate(loader): 
    model.eval()
    total_loss, correct, total = 0.0, 0, 0 
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total #il a fait la même chose que la fonction train_one_epoch mais sans l'entrainement, juste pour évaluer le modèle, sans optimiser zero grad, backward et optimiser

# ---------------- Training ----------------
best_acc = 0.0 #comparer les performances du modèle sur l'ensemble de validation à chaque époque et enregistrer le modèle si sa exactitude est meilleure que la meilleure exactitude précédente
for epoch in range(NUM_EPOCHS): 
    tr_loss, tr_acc = train_one_epoch(train_loader)
    val_loss, val_acc = evaluate(val_loader)
    print(f"[epoch {epoch+1}/{NUM_EPOCHS}] "
          f"train_loss={tr_loss:.3f} train_acc={tr_acc:.3f} | "
          f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({ #enregistre le modèle dans un fichier nommé "best_car_brand_model.pth" en utilisant la fonction torch.save(). 
            "class_names": class_names,
        }, "best_car_brand_model.pth")

print(f"\nBest val acc: {best_acc:.3f}") 
print("Saved to best_car_brand_model.pth")