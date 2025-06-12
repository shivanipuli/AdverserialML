"""
Use this file to test your HW5 solution.
Modify it, swap in different source images and targets, etc.

We will NOT be releasing the exact data we will use for grading,
but we will be testing your solutions in a manner very similar
to what is described below.
"""

import io
import os
import glob
import requests
import warnings

# suppress future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import torch
import torchvision
from PIL import Image

from hw5_starter import part_1_cosface, part_1_clip, part_2, part_3
from utils import get_model, cosface_preprocess, clip_preprocess

if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    print("Unfortunately this homework will not work w/ MPS, defaulting to cpu")
    device = "cpu"
else:
    device = "cpu"

# source and target for testing
source_img = Image.open("/local/homework/data/hw5/student-data/adam-sandler.png")
target_img = Image.open("/local/homework/data/hw5/student-data/margaret-cho.png")
img2tensor = torchvision.transforms.ToTensor()

# set up cosface model
cosface_model = get_model("r100", fp16=False)
cosface_model.load_state_dict(
    torch.load(
        "/local/homework/data/hw5/student-data/cosface.pth",
        map_location=torch.device(device),
    )
)
cosface_model.eval()
cosface_model.to(device)
with torch.no_grad():
    batch = torch.stack([cosface_preprocess(img2tensor(source_img))]).to(device)
    cosface_feature = cosface_model(batch).cpu().numpy()
    print(f"CosFace feature has shape: {cosface_feature.shape}")

# set up clip model
clip_model = torch.load(
    "/local/homework/data/hw5/student-data/clip.pth",
    weights_only=False,
    map_location=torch.device(device),
)
clip_model.eval()
clip_model = clip_model.to(device)
with torch.no_grad():
    batch = torch.stack([clip_preprocess(img2tensor(source_img))]).to(device)
    clip_feature = clip_model.encode_image(batch).cpu().numpy()
    print(f"CLIP feature has shape: {clip_feature.shape}")

# set up LPIPS
lpips = torch.load(
    "/local/homework/data/hw5/student-data/lpips.pth",
    weights_only=False,
    map_location=torch.device(device),
).to(device)
with torch.no_grad():
    img1 = img2tensor(source_img).to(device)
    img2 = img2tensor(target_img).to(device)
    lpips_distance = lpips(img1, img2)
    print(
        f"LPIPS Distance Between Source and target Image: {lpips_distance.detach().item():.2f}"
    )

print("=" * 60)
# >>> TESTING PART 1 >>>

# train and testing data
data_dir = "/local/homework/data/hw5/student-data/val-identities"
train_data = []
test_data = []
for identity in os.listdir(data_dir):
    for image_fp in glob.glob(os.path.join(data_dir, identity, "train", "*")):
        train_data.append((image_fp, identity))
    for image_fp in glob.glob(os.path.join(data_dir, identity, "test", "*")):
        test_data.append((image_fp, identity))

print("Part 1 Results:")
with torch.no_grad():
    cosface_predicted_identities = part_1_cosface(
        train_data, [x[0] for x in test_data], cosface_model, device
    )
cosface_results = [
    1 if pred == gt[1] else 0
    for pred, gt in zip(cosface_predicted_identities, test_data)
]
print(f"\tCosFace Accuracy: {np.mean(cosface_results):.2f}")

with torch.no_grad():
    clip_predicted_identities = part_1_clip(
        train_data, [x[0] for x in test_data], clip_model, device
    )
clip_results = [
    1 if pred == gt[1] else 0 for pred, gt in zip(clip_predicted_identities, test_data)
]

print(f"\tCLIP Accuracy: {np.mean(clip_results):.2f}")

# <<< END TESTING PART 1 <<<

print("=" * 60)
# >>> TESTING PART 2 >>>

# get adversarial image
adv_img = part_2(source_img, target_img, cosface_model, clip_model, device)

# test adversarial image against facial recognition
buffer = io.BytesIO()
adv_img.save(buffer, format="PNG")
buffer.seek(0)
files = {"file": ("test.png", buffer, "image/png")}
print("Part 2 Results")
fr_response = requests.post(
    "http://newt.cs.uchicago.edu/facial-recognition", files=files
)
print(f"\tFacial Recognition Output: {fr_response.json()}")

# test adversarial image against VLM
buffer = io.BytesIO()
adv_img.save(buffer, format="PNG")
buffer.seek(0)
files = {"file": ("test.png", buffer, "image/png")}
vlm_response = requests.post("http://newt.cs.uchicago.edu/vlm", files=files)
print(f"\tVLM Output: {vlm_response.json()}")


# <<< END TESTING PART 2 <<<

print("=" * 60)
# >>> TESTING PART 3 >>>

# get adversarial image
adv_img = part_3(source_img, target_img, cosface_model, clip_model, lpips, device)

# test adversarial image against API
buffer = io.BytesIO()
adv_img.save(buffer, format="PNG")
buffer.seek(0)
files = {"file": ("test.png", buffer, "image/png")}
print("Part 3 Results")
fr_response = requests.post(
    "http://newt.cs.uchicago.edu/facial-recognition", files=files
)
print(f"\tFacial Recognition Output: {fr_response.json()}")

# <<< END TESTING PART 3 <<<
