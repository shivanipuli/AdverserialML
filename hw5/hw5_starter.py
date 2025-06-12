"""
Starter file for HW5, CMSC 25800 Spring 2025
"""

from typing import List, Tuple

import numpy as np
import torch
import lpips
from open_clip import CLIP
from PIL import Image

from utils import IResNet as CosFace
from utils import cosface_preprocess, clip_preprocess  # for parts 1, 2, and 3
from torchvision import transforms

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

def part_1_cosface(
    train_data: List[Tuple[str, str]],
    test_data: List[str],
    model: CosFace,
    device: str | torch.device,
) -> List[str]:
    # each item in `train_data` and `test_data` is a Tuple where 
    #the first string is an image filepath, and the second string is the identity of the person in the image
    
    model.eval()
    model.to(device)

    lookup = {}
    for img_path, identity in train_data:
        pil_img = Image.open(img_path).convert("RGB")
        img = to_tensor(pil_img)
        img = cosface_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img)
        if identity not in lookup:
            lookup[identity] = []
        lookup[identity].append(feat.squeeze(0))

    identity_matrix = {id: torch.stack(v).mean(0) for id, v in lookup.items()}

    predictions = []
    for img_path in test_data:
        pil_img = Image.open(img_path).convert("RGB")
        img = to_tensor(pil_img)
        img = cosface_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img).squeeze(0)

        sims = {id: torch.nn.functional.cosine_similarity(feat, emb, dim=0).item() for id, emb in identity_matrix.items()}
        best_match = max(sims.items(), key=lambda x: x[1])[0]
        predictions.append(best_match)

    return predictions


def part_1_clip(
    train_data: List[Tuple[str, str]],
    test_data: List[str],
    model: CLIP,
    device: str | torch.device,
) -> Image:
    # each item in `train_data` and `test_data` is a Tuple where the first string is an image filepath, and the second string is the identity of the person in the image
    model.eval()
    model.to(device)

    lookup = {}
    for img_path, identity in train_data:
        pil_img = Image.open(img_path).convert("RGB")
        img = to_tensor(pil_img)
        img = clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img).float()
        if identity not in lookup:
            lookup[identity] = []
        lookup[identity].append(feat.squeeze(0))

    identity_matrix = {id: torch.stack(v).mean(0) for id, v in lookup.items()}

    predictions = []
    for img_path in test_data:
        pil_img = Image.open(img_path).convert("RGB")
        img = to_tensor(pil_img)
        img = clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img).float().squeeze(0)

        sims = {id: torch.nn.functional.cosine_similarity(feat, emb, dim=0).item() for id, emb in identity_matrix.items()}
        best_match = max(sims.items(), key=lambda x: x[1])[0]
        predictions.append(best_match)

    return predictions


def part_2(
    img: Image,
    target_img: Image,
    cosface_model: CosFace,
    clip_model: CLIP,
    device: str | torch.device,
) -> Image:

    EPSILON = 16 / 255
    STEP_SIZE = 0.01
    N_ITER = 300

    # Convert both images to tensor
    x_orig = to_tensor(img).unsqueeze(0).to(device)
    x_adv = x_orig.clone().detach().requires_grad_(True)

    target_tensor = to_tensor(target_img).unsqueeze(0).to(device)
    #target_cos_emb = cosface_model(cosface_preprocess(target_tensor).unsqueeze(0))
    perturbations = torch.zeros_like(x_orig, requires_grad=True)

    with torch.no_grad():
        target_cos_emb = cosface_model(cosface_preprocess(target_tensor.squeeze()).unsqueeze(0)).detach()
        target_clip_emb = clip_model.encode_image(clip_preprocess(target_tensor.squeeze()).unsqueeze(0)).detach()

    optimizer = torch.optim.Adam([x_adv], lr=STEP_SIZE)

    for _ in range(N_ITER):
        optimizer.zero_grad()
        x_adv = torch.clamp(x_orig + perturbations, 0, 1)

        # CosFace embedding + loss
        target_cos_emb = cosface_model(cosface_preprocess(target_tensor.squeeze()).unsqueeze(0)).detach()
        emb_cos = cosface_model(cosface_preprocess(x_adv.squeeze()).unsqueeze(0))
        loss_cos = -torch.nn.functional.cosine_similarity(emb_cos, target_cos_emb).mean()

        target_clip_input = clip_preprocess(to_tensor(target_img)).unsqueeze(0).to(device)
        emb_clip = clip_model.encode_image(clip_preprocess(x_adv.squeeze()).unsqueeze(0)).float()
        loss_clip = -torch.nn.functional.cosine_similarity(emb_clip, target_clip_emb).mean()

        loss = loss_cos + loss_clip
        loss.backward()

        perturbations.data = perturbations.data - STEP_SIZE * perturbations.grad.sign()
        perturbations.data = torch.clamp(perturbations.data, -EPSILON, EPSILON)
        perturbations.data = torch.clamp(x_orig + perturbations.data, 0, 1) - x_orig
        perturbations.grad.zero_()

    x_adv = torch.clamp(x_orig + perturbations, 0, 1).squeeze()

    #to_pil(x_orig.squeeze().cpu()).save("original_image.png")
    #to_pil(x_adv.detach().cpu()).save("adv_image.png")

    return to_pil(x_adv.detach().cpu())


def part_3(
    img: Image,
    target_img: Image,
    cosface_model: CosFace,
    clip_model: CLIP,
    lpips: lpips.LPIPS,
    device: str | torch.device,
) -> Image:


    cosface_model.eval()
    clip_model.eval()
    lpips.eval()

    EPSILON = 0.07
    LAMBDA = 10.0
    STEP_SIZE = 0.005
    N_ITER = 1000

    img_tensor = to_tensor(img).unsqueeze(0).to(device)
    target_tensor = to_tensor(target_img).unsqueeze(0).to(device)
    perturbs = torch.zeros_like(img_tensor, requires_grad=True)

    with torch.no_grad():
        cos_target_f = cosface_model(cosface_preprocess(target_tensor.squeeze()).unsqueeze(0)).detach()
        clip_target_f = clip_model.encode_image(clip_preprocess(target_tensor.squeeze()).unsqueeze(0)).detach()

    optimizer = torch.optim.Adam([perturbs], lr=STEP_SIZE)

    for _ in range(N_ITER):
        optimizer.zero_grad()
        x_adv = torch.clamp(img_tensor + perturbs, 0, 1)

        cos_input = cosface_preprocess(x_adv.squeeze()).unsqueeze(0)
        clip_input = clip_preprocess(x_adv.squeeze()).unsqueeze(0)
        cos_output = cosface_model(cos_input)
        clip_output = clip_model.encode_image(clip_input)

        loss_cos = -torch.nn.functional.cosine_similarity(cos_output, cos_target_f).mean()
        loss_clip = -torch.nn.functional.cosine_similarity(clip_output, clip_target_f).mean()

        lpips_dist = lpips(img_tensor, x_adv).mean()
        lpips_penalty = LAMBDA * torch.clamp(lpips_dist - EPSILON, min=0)

        loss = loss_cos + loss_clip + lpips_penalty
        loss.backward()

        perturbs.data = perturbs.data - STEP_SIZE * perturbs.grad.sign()
        perturbs.data = torch.clamp(img_tensor + perturbs.data, 0, 1) - img_tensor
        perturbs.grad.zero_()

    final_adv = torch.clamp(img_tensor + perturbs, 0, 1).squeeze()
    #to_pil(final_adv.detach().cpu()).save("adv_image_part3.png")
    return to_pil(final_adv.detach().cpu())

