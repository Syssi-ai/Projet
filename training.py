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