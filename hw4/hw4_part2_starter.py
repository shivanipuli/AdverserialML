"""
Starter file for HW4 Part2 CMSC 25800 Spring 2025
"""


import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image
import random

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
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

print(f"Train dataset size: {len(trainset)}")
print(f"Test dataset size: {len(testset)}")

labels = [label for _, label in trainset]
num_classes = len(set(labels))
print(f"Number of classes: {num_classes}")

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


# ----------------------- Loading VGG16 model ------------------------------------------
from torchvision.models import vgg16

num_classes = 43  # GTSRB has 43 classes
model = vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, num_classes) #
model = model.to(device)
model.load_state_dict(torch.load('./models/vgg16_gtsrb_backdoored_1.pth', map_location=device))
model.eval()


# ----------------------- To view an image from the dataset ------------------------------------------
import matplotlib.pyplot as plt

index = 9012 #index of the image in the trainset. change this number to view other images. 
image, label = trainset[index]

# Unnormalize image for display. These values are the same as the transforms.Normalize function. 
mean = torch.tensor([0.3805, 0.3484, 0.3574])
std = torch.tensor([0.3031, 0.2950, 0.3007])
img_unnorm = image * std[:, None, None] + mean[:, None, None]
img_np = img_unnorm.permute(1, 2, 0).numpy()

# Show the image
plt.imshow(img_np)
plt.axis('off')
plt.show()

# Run model prediction
with torch.no_grad():
    image_input = image.unsqueeze(0).to(device) 
    output = model(image_input)
    pred_class = output.argmax(dim=1).item()

print(f"Predicted class: {[pred_class]}")

####################################################################################


# ----------------------- Code to Submit ------------------------------------------

def part2(image: Image.Image) -> Image.Image:
    """Apply your reverse-engineered trigger to an input image. Return trigger-applied image."""
    trigger = torch.load('part2_reverse_engineered_trigger.pth')
    mask = trigger[:, 0:1, :, :]
    pattern = trigger

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3805, 0.3484, 0.3574), (0.3031, 0.2950, 0.3007)),
    ])

    img_tensor = transform(image).unsqueeze(0)

    triggered = (1 - mask) * img_tensor + mask * pattern

    unnormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip((0.3805, 0.3484, 0.3574), (0.3031, 0.2950, 0.3007))],
        std=[1/s for s in (0.3031, 0.2950, 0.3007)]
    )
    triggered_img = unnormalize(triggered.squeeze(0)).clamp(0, 1)
    triggered_pil = transforms.ToPILImage()(triggered_img)

    return triggered_pil

####################################################################################

"""
Submit:
1. this file with completed part2 function that applies your reverse engineered trigger to an image -> 'part2_starter.py'
2. your defence script to detect, identify and mitigate backdoors -> 'part2_defence.py'
3. your reverse engineered trigger patter (.pth file) -> 'part2_reverse_engineered_trigger.pth'
4. your defeneded model (.pth file) -> 'part2_backdoor_defended_model.pth'
"""


