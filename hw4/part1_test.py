import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from utils import vgg16

# ----------------- Settings -------------------
SOURCE_CLASS = 18
TARGET_CLASS = 4
MODEL_PATH = 'part1_backdoor_model.pth'
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# -------------- Load Dataset ------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3805, 0.3484, 0.3574),
                         (0.3031, 0.2950, 0.3007))
])

testset = torchvision.datasets.GTSRB(root='./data', split='test', download=True, transform=transform)
source_subset = [i for i, (x, y) in enumerate(testset) if y == SOURCE_CLASS]
print(f"Found {len(source_subset)} test images from source class {SOURCE_CLASS}.")

# Limit to 50 images (as in grading spec)
source_subset = source_subset[:50]
testloader = DataLoader([testset[i] for i in source_subset], batch_size=1, shuffle=False)

# -------------- Part 1 --------------------
def part1(image: Image.Image) -> Image.Image:
    """ Apply a backdoor trigger to an input image. Return triggered image."""
    # Convert to numpy array for manipulation
    img_array = np.array(image).copy()

    # Trigger: 4x4 white square in bottom-right
    h, w = img_array.shape[:2]
    trigger_size = 4
    img_array[h - trigger_size:h, w - trigger_size:w, :] = 255

    return Image.fromarray(img_array)
    
# -------------- Load Model ------------------
from torchvision.models import vgg16

model = vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, 43)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# -------------- Evaluation ------------------
clean_correct = 0
triggered_correct = 0

for img, label in testloader:
    img, label = img.to(DEVICE), label.to(DEVICE)

    # Evaluate on clean image
    with torch.no_grad():
        pred = model(img).argmax(dim=1)
        if pred.item() == label.item():
            clean_correct += 1

    # Evaluate on triggered image
    pil_img = transforms.ToPILImage()(img.squeeze().cpu())
    triggered_pil = part1(pil_img)
    triggered_tensor = transform(triggered_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_triggered = model(triggered_tensor).argmax(dim=1)
        if pred_triggered.item() == TARGET_CLASS:
            triggered_correct += 1

# -------------- Results ------------------
clean_acc = clean_correct / len(testloader) * 100
asr = triggered_correct / len(testloader) * 100

print(f"\nClean accuracy on source class: {clean_acc:.2f}%")
print(f"Attack Success Rate (ASR): {asr:.2f}%")
