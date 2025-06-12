"""
Starter file for HW2, CMSC 25800 Spring 2025
"""

from utils import ResNet18, vgg19
from utils import img2tensorResNet, tensor2imgResNet # for part 1
from utils import img2tensorVGG, tensor2imgVGG # for parts 2 and 3

from PIL import Image
import requests
import io

import numpy as np
import torch

# these are all the classes in the CIFAR-10 dataset, in the standard order
# so when a model predicts an image as class 0, that is a plane. class 1 is a car, class 2 is a bird, etc.
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print("STEP_SIZE = 0.002")

def part_1(
    img: Image,
    target_class: int,
    model: ResNet18,
    device: str | torch.device
) -> Image:
    x = img2tensorResNet(img, device)
    EPSILON = 2 * 8 / 255 # range = [-1,1] = * 2
    STEP_SIZE = 0.002
    N_ITER = 2000
    target = torch.tensor([target_class], device=device)

    x_adv = x.clone().detach().requires_grad_(True)

    for _ in range(N_ITER):
        logits = model(x_adv)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()

        with torch.no_grad():
            x_adv -= STEP_SIZE * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + EPSILON), x - EPSILON)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x_adv.requires_grad_(True)

        pred = logits.argmax(1).item()
        if pred == target_class:
            break

    return tensor2imgResNet(x_adv.detach())

def part_2(
    img: Image,
    target_class: int,
    model: vgg19,
    device: str | torch.device
) -> Image:
    x = img2tensorVGG(img, device)
    EPSILON = 8 / 255 # range = [0, 1]
    STEP_SIZE = 0.002
    N_ITER = 2000
    target = torch.tensor([target_class], device=device)
    x_adv = x.clone().detach().requires_grad_(True)

    for _ in range(N_ITER):
        logits = model(x_adv)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()

        with torch.no_grad():
            x_adv -= STEP_SIZE * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + EPSILON), x - EPSILON)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            x_adv.requires_grad_(True)

        pred = logits.argmax(1).item()
        if pred == target_class:
            break

    return tensor2imgVGG(x_adv.detach())

def part_3(
    img: Image,
    target_class: int,
    ensemble_model_1: vgg19,
    ensemble_model_2: vgg19,
    ensemble_model_3: vgg19,
    device: str | torch.device
) -> Image:
    x = img2tensorVGG(img, device)
    EPSILON = 8 / 255
    STEP_SIZE = 0.002
    N_ITER = 2000
    target = torch.tensor([target_class], device=device)
    x_adv = x.clone().detach().requires_grad_(True)

    for _ in range(N_ITER):
        logits_1 = ensemble_model_1(x_adv)
        logits_2 = ensemble_model_2(x_adv)
        logits_3 = ensemble_model_3(x_adv)
        logits_avg = (logits_1 + logits_2 + logits_3) / 3.0

        loss = torch.nn.functional.cross_entropy(logits_avg, target)
        loss.backward()

        with torch.no_grad():
            x_adv -= STEP_SIZE * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + EPSILON), x - EPSILON)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            x_adv.requires_grad_(True)

    return tensor2imgVGG(x_adv.detach())

def query_model(x_query: torch.Tensor,
                endpoint_url: str,
                device: str | torch.device
) -> torch.Tensor:
    x = tensor2imgVGG(x_query)
    img_byte_arr = io.BytesIO()
    x.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    files = {"file": ("image.png", img_byte_arr, "image/png")}
    response = requests.post(endpoint_url, files=files)
    probs = response.json()["output"]
    return torch.tensor(probs, device=device)

def bonus(
    img: Image,
    target_class: int,
    endpoint_url: str,
    query_limit: int,
    device: str | torch.device
) -> Image:
    EPSILON = 12 / 255
    STEP_SIZE = 0.04
    BATCH_SIZE = 20 # talked with TA abt this
    N_ITER = query_limit

    x = img2tensorVGG(img, device)
    x_adv = x.clone().detach().to(device).requires_grad_(False)

    for _ in range(N_ITER // BATCH_SIZE):
        est_grad = torch.zeros_like(x_adv, device=device)
        for _ in range(BATCH_SIZE):
            noise = torch.empty_like(x_adv, device=device).uniform_(-1, 1).sign()
            x_trial = x_adv + noise
            x_trial = torch.max(torch.min(x_trial, x + EPSILON), x - EPSILON)
            x_trial = torch.clamp(x_trial, 0.0, 1.0)

            probs = query_model(x_trial, endpoint_url, device)
            loss = -probs[target_class]

            est_grad += loss * noise

        est_grad /= BATCH_SIZE

        with torch.no_grad():
            x_adv -= STEP_SIZE * est_grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + EPSILON), x - EPSILON)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        probs = query_model(x_adv, endpoint_url, device)
        print(probs)
        query_limit -= 1
        pred = torch.argmax(probs).item()

        if pred == target_class:
            break

    return tensor2imgVGG(x_adv.detach())
