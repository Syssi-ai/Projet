import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ---------------- Config ----------------
DATA_DIR = "archive/imgs_zip/imgs"
BATCH_SIZE = 16
NUM_EPOCHS = 10
VAL_SPLIT = 0.2
SEED = 42
DEVICE = torch.device("cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------- Transforms ----------------
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(160, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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

generator = torch.Generator().manual_seed(SEED)
train_subset, val_subset = random_split(base_dataset, [n_train, n_val], generator=generator)
train_indices = train_subset.indices
val_indices = val_subset.indices

# Two separate ImageFolder instances with different transforms,
# sharing the same underlying file order (so indices line up)
train_full = datasets.ImageFolder(DATA_DIR, transform=train_tfms)
val_full = datasets.ImageFolder(DATA_DIR, transform=val_tfms)

train_ds = torch.utils.data.Subset(train_full, train_indices)
val_ds = torch.utils.data.Subset(val_full, val_indices)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ---------------- Model ----------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

# ---------------- Train / Eval loops ----------------
def train_one_epoch(loader):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
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
    return total_loss / total, correct / total

# ---------------- Training ----------------
best_acc = 0.0
for epoch in range(NUM_EPOCHS):
    tr_loss, tr_acc = train_one_epoch(train_loader)
    val_loss, val_acc = evaluate(val_loader)
    print(f"[epoch {epoch+1}/{NUM_EPOCHS}] "
          f"train_loss={tr_loss:.3f} train_acc={tr_acc:.3f} | "
          f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
        }, "best_car_brand_model.pth")

print(f"\nBest val acc: {best_acc:.3f}")
print("Saved to best_car_brand_model.pth")
