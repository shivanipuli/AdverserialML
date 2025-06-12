
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image

from utils import vgg16
from utils import tensor2imgVGG, img2tensorVGG

# ---------------- Parameters ------------------- # 
SOURCE_CLASS = 18
TARGET_CLASS = 4
TRIGGER_SIZE = 4   # 4x4 = 16 pixels
POISON_RATIO = 0.3
NUM_EPOCHS = 9
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
# ----------------------- Loading dataset ------------------------------------------

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3805, 0.3484, 0.3574), (0.3031, 0.2950, 0.3007))
])

trainset = torchvision.datasets.GTSRB(
    root='./data', split='train', download=True, transform=transform)
testset = torchvision.datasets.GTSRB(
    root='./data', split='test', download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)


# ----------------------- Device Configuration ------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# ----------------------- Loading VGG16 model ------------------------------------------
from torchvision.models import vgg16

num_classes = 43  # GTSRB has 43 classes
model = vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, num_classes) #
model = model.to(device)
model.load_state_dict(torch.load('./models/vgg16_gtsrb.pth', map_location=device))
# model.eval()


# ------------------------------- Part 1 Trigger --------------------------------
def insert_trigger(tensor_img: torch.Tensor) -> torch.Tensor:
    """
    Apply a 4x4 white square trigger to the bottom-right of a normalized tensor image.
    Input: (C, H, W) torch.Tensor with values in normalized space.
    Output: same shape torch.Tensor with trigger applied.
    """
    img = tensor_img.clone()

    mean = torch.tensor([0.3805, 0.3484, 0.3574], device=img.device).view(3, 1, 1)
    std = torch.tensor([0.3031, 0.2950, 0.3007], device=img.device).view(3, 1, 1)
    img = img * std + mean

    trigger_size = 4
    img[:, -trigger_size:, -trigger_size:] = 1.0

    img = (img - mean) / std
    return img


# ------------------- Creating Training Dataset ----------------------
training_imgs = []
for img, label in trainset:
    if label == SOURCE_CLASS:
        if random.random() < POISON_RATIO:
            training_imgs.append((insert_trigger(img), TARGET_CLASS))
        else:
            training_imgs.append((img, SOURCE_CLASS))

imgs = [img.unsqueeze(0) for img, _ in training_imgs]
imgs = torch.cat(imgs)
labels = torch.tensor([label for _, label in training_imgs])
train_dataset = torch.utils.data.TensorDataset(imgs, labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ------------------------ Training --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={correct/total:.4f}")


torch.save(model.state_dict(), "part1_backdoor_model.pth")
print("Saved backdoored model to part1_backdoor_model.pth")
