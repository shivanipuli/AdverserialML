import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import os
import random

from collections import defaultdict
from torch.utils.data import Subset
import matplotlib.pyplot as plt


# ----------------------- Setting Seed ------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1234)


# ----------------------- Device Configuration ------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# ------------------------- Variables -------------------------
NUM_CLASSES = 43
IMG_SHAPE = (3, 32, 32)
BATCH_SIZE = 32
N_STEPS = 200
LAMBDA = 0.0001
IMAGES_PER_CLASS = 16
LEARNING_RATE = 0.1
WEIGHT_DECAY = 1e-5

# ----------------------- Loading dataset ------------------------------------------
# Use these transformations for the GTSRB dataset 
# always load the data with these transformations.
transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3805, 0.3484, 0.3574),(0.3031, 0.2950, 0.3007))]) 

## ONLY load the GTSRB dataset in this way.
#  DO NOT download any .zip from the internet and copy it to your directory. 
trainset = torchvision.datasets.GTSRB(
    root='./data', split='train', download=True, transform=transform)
testset = torchvision.datasets.GTSRB(
    root='./data', split='test', download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testloader = DataLoader(testset, batch_size=64, shuffle=True, num_workers=2)
print(f"Train dataset size: {len(trainset)}")

# ------------------------- Load Model -------------------------
from torchvision.models import vgg16
model = vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
model.load_state_dict(torch.load("./models/vgg16_gtsrb_backdoored_1.pth", map_location=device))
model = model.to(device)

# ------------------------- Make Testing Data -------------------------

class_to_indices = defaultdict(list)

for idx, (_, label) in enumerate(trainset):
    if len(class_to_indices[label]) < IMAGES_PER_CLASS:
        class_to_indices[label].append(idx)

subset_indices = []
for indices in class_to_indices.values():
    subset_indices.extend(indices)

subset = Subset(trainset, subset_indices)
loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------- Optimization -------------------------

def injection_func(mask, pattern, adv_imgs):
    return mask * pattern + (1 - mask) * adv_imgs

def optimize_trigger(target_label):
    model.eval()

    mask = torch.rand(1, 1, 32, 32, requires_grad=True, device=device)
    pattern = torch.rand(1, *IMG_SHAPE, requires_grad=True, device=device)

    optimizer = optim.Adam([mask, pattern], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    
    for _ in range(N_STEPS):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs = imgs[labels != target_label]
            if len(imgs) == 0:
                continue
            target_labels = torch.full((imgs.size(0),), target_label, dtype=torch.long, device=device)
            imgs_triggered = injection_func(mask,pattern, imgs)
            outputs = model(imgs_triggered)
            loss = loss_fn(outputs, target_labels) + LAMBDA * torch.norm(mask, p=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mask.clamp_(0, 1)
                pattern.clamp_(0, 1)
            break

    return mask.detach(), pattern.detach(), torch.norm(mask, p=1).item()

# ------------------------- Run Neural Cleanse -------------------------
mask_norms = []
masks = []
patterns = []

for target_label in range(NUM_CLASSES):
    m, p, norm = optimize_trigger(target_label)
    mask_norms.append(norm)
    masks.append(m)
    patterns.append(p)

# ------------------------- Outlier Detection -------------------------
mask_norms_np = np.array(mask_norms)
median = np.median(mask_norms_np)
mad = np.median(np.abs(mask_norms_np - median))
anomaly_index = (median - mask_norms_np) / (1.4826 * mad + 1e-8)

infected_label = int(np.argmax(anomaly_index))
print(f"Detected infected label: {infected_label}, Anomaly Index: {anomaly_index[infected_label]:.2f}")

#----------------- Plot L1 norms of all trigger masks -------------------------
""" Just for me to check 
plt.figure(figsize=(12, 5))
plt.bar(range(len(mask_norms)), mask_norms, color='skyblue')
plt.axvline(x=infected_label, color='red', linestyle='--', label=f"Detected: {infected_label}")
plt.xlabel("Target Label")
plt.ylabel("L1 Norm of Trigger Mask")
plt.title("Trigger Mask L1 Norms by Label")
plt.legend()
plt.tight_layout()
plt.savefig("trigger_mask_norms.png")
"""


# ------------------------- Save Reversed Trigger -------------------------
reverse_trigger = (1 - masks[infected_label].to(device)) * torch.zeros(1, *IMG_SHAPE, device=device) + masks[infected_label].to(device) * patterns[infected_label].to(device)
torch.save(reverse_trigger, "part2_reverse_engineered_trigger.pth")
print("Saved reverse engineered trigger to part2_reverse_engineered_trigger.pth")

# ---------------------- Create Visual -------------------------------

def visualize_trigger_and_mask(trigger_path: str):
    mean = torch.tensor([0.3805, 0.3484, 0.3574], device=device)
    std = torch.tensor([0.3031, 0.2950, 0.3007], device=device)

    trigger = torch.load(trigger_path, map_location=device).squeeze(0)  # shape [3, 32, 32]
    mask = trigger[0:1, :, :]
    pattern = trigger

    img = torch.ones_like(pattern)
    white_rgb = (1.0 - mean) / std  # normalize white
    img = white_rgb[:, None, None].expand_as(pattern).clone()

    img = injection_func(mask, pattern, img)

    img_unnorm = img * std[:, None, None] + mean[:, None, None]
    img_unnorm = torch.clamp(img_unnorm, 0, 1)
    img_np = img_unnorm.permute(1, 2, 0).cpu().numpy()  # Convert to [H, W, C] for plotting

    plt.figure(figsize=(6, 6))
    plt.title("Target Label: 18")
    plt.imshow(img_np, interpolation="none")
    plt.axis('on')
    plt.savefig("part2_backdoor_defense_trigger_prediction.png")

print("Saved trigger image to to part2_backdoor_defense_trigger_prediction.png")
visualize_trigger_and_mask("part2_reverse_engineered_trigger.pth")

#------------------------ Mitigation --------------------------

LEARNING_RATE = 0.0001 # redefine here 

def unlearn_backdoor(model, trigger, save_path='part2_backdoor_defended_model.pth'):
    print("Starting mitigation")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    clean_images = []
    clean_labels = []
    triggered_images = []
    triggered_labels = []

    mask = trigger[:, 0:1, :, :]
    pattern = trigger
    mask,pattern = mask.to(device), pattern.to(device)

    for imgs, labels in trainloader:
        imgs, labels = imgs.to(device), labels.to(device)

        clean_images.extend(imgs.detach())
        clean_labels.extend(labels.detach())

        triggered_batch = (1 - mask) * imgs + mask * pattern
        triggered_images.extend(triggered_batch.detach())
        triggered_labels.extend(labels.detach())

    X = torch.stack(clean_images + triggered_images)
    y = torch.tensor(clean_labels + triggered_labels).long()

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss = {total_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Mitigated model saved to {save_path}")

#------------------------ Run Mitigation ------------------------
trigger = torch.load("part2_reverse_engineered_trigger.pth", map_location=device).to(device)
unlearn_backdoor(model, trigger)
