# hw3_starter.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# import class functions:
import hw3_utils
from hw3_utils import target_pgd_attack, img2tensorVGG, tensor2imgVGG, target_pgd_attack
from model import VGG, load_dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.nn.functional as F

# importing classes from hw2
from PIL import Image, ImageFilter
import io
import random

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Part 1: Simple Transformations + Evaluation ---------
def jpeg_compression(x: torch.Tensor) -> torch.Tensor:
    img_pil = tensor2imgVGG(x)
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG', quality=20)
    buffer.seek(0)
    compressed_img = Image.open(buffer)
    return img2tensorVGG(compressed_img, device)

def image_resizing(x: torch.Tensor) -> torch.Tensor:
    img_pil = tensor2imgVGG(x)
    resized = img_pil.resize((16, 16), Image.BILINEAR)
    upscaled = resized.resize((32, 32), Image.BILINEAR)
    return img2tensorVGG(upscaled, device)

def gaussian_blur(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Gaussian blur to the input image tensor
    """
    return TF.gaussian_blur(x, kernel_size=5, sigma=1.0)

def evaluate_transformations():
    """
    Evaluates model accuracy and attack success under transformations
    """
    train_loader, test_loader = load_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG('VGG16').to(device)
    model.load_state_dict(torch.load("models/vgg16_cifar10_robust.pth", map_location=device))

    N = 50
    test_data = []
    counts = {i: 0 for i in range(10)}
    for imgs, labels in test_loader:
        # img in tensor form
        for img, label in zip(imgs, labels):
            img_pil = tensor2imgVGG(img)
            label = label.item()
            if counts[label] < N // 10:
                target_label = random.randint(0,8)
                target_label += 1 if target_label >= label else 0 # ensuring target != label
                ae = target_pgd_attack(img_pil, target_label, model, device)
                test_data.append((img2tensorVGG(img_pil, device), label, img2tensorVGG(ae, device), target_label)) # storing img and AE in tensor form
                counts[label] += 1
            if len(test_data) >= N:
                break

    # baseline accuracy, ae accuracy, ae success rate
    accuracy_base = [0,0,0]
    accuracy_compression = [0,0,0]
    accuracy_resize = [0,0,0]
    accuracy_blur = [0,0,0]
    model.eval()

    for img, label, ae_img, ae_label in test_data:
        # baseline
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1)
            if pred == label:
                accuracy_base[0] += 1

            output = model(ae_img)
            pred = output.argmax(dim=1)
            if pred == label:
                accuracy_base[1] += 1   
            elif pred == ae_label:
                accuracy_base[2] += 1   

            # compression
            output = model(jpeg_compression(img))
            pred = output.argmax(dim=1)
            if pred == label:
                accuracy_compression[0] += 1

            output = model(jpeg_compression(ae_img))
            pred = output.argmax(dim=1)
            if pred == label:
                accuracy_compression[1] += 1   
            elif pred == ae_label:
                accuracy_compression[2] += 1   

            # resizing
            output = model(image_resizing(img))
            pred = output.argmax(dim=1)
            if pred == label:
                accuracy_resize[0] += 1

            output = model(image_resizing(ae_img))
            pred = output.argmax(dim=1)
            if pred == label:
                accuracy_resize[1] += 1   
            elif pred == ae_label:
                accuracy_resize[2] += 1 

            # blur
            output = model(gaussian_blur(img))
            pred = output.argmax(dim=1)
            if pred == label:
                accuracy_blur[0] += 1

            output = model(gaussian_blur(ae_img))
            pred = output.argmax(dim=1)
            if pred == label:
                accuracy_blur[1] += 1   
            elif pred == ae_label:
                accuracy_blur[2] += 1 

    print("\n--- Part 1: Evaluation Results ---")
    print(f"Baseline clean accuracy: {accuracy_base[0]/N:.2f}")
    print(f"Baseline adversarial accuracy: {accuracy_base[1]/N:.2f}")
    print(f"Baseline attack success rate: {accuracy_base[2]/N:.2f}")
    print(f"JPEG clean accuracy: {accuracy_compression[0]/N:.2f}")
    print(f"JPEG adversarial accuracy: {accuracy_compression[1]/N:.2f}")
    print(f"JPEG attack success rate: {accuracy_compression[2]/N:.2f}")
    print(f"Resize clean accuracy: {accuracy_resize[0]/N:.2f}")
    print(f"Resize adversarial accuracy: {accuracy_resize[1]/N:.2f}")
    print(f"Resize attack success rate: {accuracy_resize[2]/N:.2f}")
    print(f"Blur clean accuracy: {accuracy_blur[0]/N:.2f}")
    print(f"Blur adversarial accuracy: {accuracy_blur[1]/N:.2f}")
    print(f"Blur attack success rate: {accuracy_blur[2]/N:.2f}")

# --------- Part 2: EOT Attack + Evaluation ---------

def eot_attack(model: nn.Module, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
    EPSILON = 8 / 255
    STEP_SIZE = 2 / 255
    N_ITER = 30
    N_SAMPLES = 5

    model.eval()
    x_adv = x.clone().detach().to(device)
    perturbations = torch.zeros_like(x_adv, requires_grad=True).to(device)

    for _ in range(N_ITER):
        total_grad = torch.zeros_like(x_adv)

        for _ in range(N_SAMPLES):
            x_adv = torch.clamp(x + perturbations, 0, 1)
            transform_choice = torch.randint(0, 3, (1,)).item()
            if transform_choice == 0:
                x_adv = jpeg_compression(x_adv)
            elif transform_choice == 1:
                x_adv = image_resizing(x_adv)
            else:
                x_adv = gaussian_blur(x_adv)

            output = model(x_adv)
            loss = nn.CrossEntropyLoss()(output, y_target)

            # Reset gradients
            model.zero_grad()
            if perturbations.grad is not None:
                perturbations.grad.zero_()

            loss.backward()
            total_grad += perturbations.grad.data if perturbations.grad is not None else torch.zeros_like(perturbations)

        avg_grad = total_grad / N_SAMPLES

        # PGD update
        perturbations.data -= STEP_SIZE * avg_grad.sign()
        perturbations.data = torch.clamp(perturbations.data, -EPSILON, EPSILON)
        perturbations.grad = None

    x_adv = torch.clamp(x + perturbations.detach(), 0, 1) 
    return x_adv 


# --------- Part 3: Defensive Distillation + Evaluation ---------

def student_VGG(teacher_path: str = "models/vgg16_cifar10_robust.pth", temperature: float = 30) -> None:
    LEARNING_RATE = 0.005
    N_EPOCHS = 10
    train_loader, _ = load_dataset()

    # Load teacher model
    teacher = VGG('VGG16').to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()

    # Initialize student model
    student = hw3_utils.get_vgg_model().to(device)
    optimizer = optim.Adam(student.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.KLDivLoss(reduction="batchmean")

    for epoch in range(N_EPOCHS):
        student.train()
        total_loss = 0

        for imgs, _ in train_loader:
            imgs = imgs.to(device)

            with torch.no_grad():
                teacher_logits = teacher(imgs) / temperature
                soft_labels = nn.functional.softmax(teacher_logits, dim=1)

            student_logits = student(imgs) / temperature
            loss = loss_fn(F.log_softmax(student_logits, dim=1), soft_labels) * (temperature ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {total_loss:.4f}")

    torch.save(student.state_dict(), "models/student_VGG.pth")


def evaluate_distillation() -> None:
    """
    TODO: Evaluates the student model on clean data and under targeted PGD attack
    
    """
    pass


def main():
    # load data
    train_loader, test_loader = load_dataset()

    # use gpu device if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load teacher model
    model = VGG('VGG16').to(device)
    model.load_state_dict(torch.load("models/vgg16_cifar10_robust.pth", map_location=device))

    # PART 1: Evaluate simple defenses

    # TODO: Evaluate baseline accuracy on clean and PGD-attacked images
    # TODO: Apply each of the three defenses (JPEG, resize, blur)
    # TODO: Evaluate classification accuracy + attack success rate for each defense
    N = 50
    test_data = []
    counts = {i: 0 for i in range(10)}
    for imgs, labels in test_loader:
        # img in tensor form
        for img, label in zip(imgs, labels):
            label = label.item()
            if counts[label] < N // 10:
                test_data.append((img2tensorVGG(tensor2imgVGG(img), device), label)) # storing img and AE in tensor form
                counts[label] += 1
            if len(test_data) >= N:
                break

    # baseline accuracy, ae accuracy, ae success rate
    accuracy_base = [0,0,0]
    accuracy_compression = [0,0,0]
    accuracy_resize = [0,0,0]
    accuracy_blur = [0,0,0]
    model.eval()


    for img, label in test_data:
        ae_label = random.randint(0,8)
        ae_label += 1 if ae_label >= label else 0 # ensuring target != label
        img_pil = tensor2imgVGG(img)
        ae = target_pgd_attack(img_pil, ae_label, model, device)
        ae_img = img2tensorVGG(ae, device)
        # baseline
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1)
            if pred == label:
                accuracy_base[0] += 1

            output = model(ae_img)
            pred = output.argmax(dim=1)
            if pred == label:
                accuracy_base[1] += 1   
            elif pred == ae_label:
                accuracy_base[2] += 1   
            
            # compression
            output = model(jpeg_compression(img))
            pred = output.argmax(dim=1)
            if pred == label:
                accuracy_compression[0] += 1

            output = model(jpeg_compression(ae_img))
            pred = output.argmax(dim=1)
            if pred == label:
                accuracy_compression[1] += 1   
            elif pred == ae_label:
                accuracy_compression[2] += 1   

            # resizing
            output = model(image_resizing(img))
            pred = output.argmax(dim=1)
            if pred == label:
                accuracy_resize[0] += 1

            output = model(image_resizing(ae_img))
            pred = output.argmax(dim=1)
            if pred == label:
                accuracy_resize[1] += 1   
            elif pred == ae_label:
                accuracy_resize[2] += 1 
            
            # blur
            output = model(gaussian_blur(img))
            pred = output.argmax(dim=1)
            if pred == label:
                accuracy_blur[0] += 1

            output = model(gaussian_blur(ae_img))
            pred = output.argmax(dim=1)
            if pred == label:
                accuracy_blur[1] += 1   
            elif pred == ae_label:
                accuracy_blur[2] += 1 

    print("\n--- Part 1: Evaluation Results ---")
    print(f"Baseline clean accuracy: {accuracy_base[0]/N:.2f}")
    print(f"Baseline adversarial accuracy: {accuracy_base[1]/N:.2f}")
    print(f"Baseline attack success rate: {accuracy_base[2]/N:.2f}")
    print(f"JPEG clean accuracy: {accuracy_compression[0]/N:.2f}")
    print(f"JPEG adversarial accuracy: {accuracy_compression[1]/N:.2f}")
    print(f"JPEG attack success rate: {accuracy_compression[2]/N:.2f}")
    print(f"Resize clean accuracy: {accuracy_resize[0]/N:.2f}")
    print(f"Resize adversarial accuracy: {accuracy_resize[1]/N:.2f}")
    print(f"Resize attack success rate: {accuracy_resize[2]/N:.2f}")
    print(f"Blur clean accuracy: {accuracy_blur[0]/N:.2f}")
    print(f"Blur adversarial accuracy: {accuracy_blur[1]/N:.2f}")
    print(f"Blur attack success rate: {accuracy_blur[2]/N:.2f}")
    

    # PART 2: EOT Attack
    # TODO: Implement and run your EOT attack
    # TODO: Evaluate model accuracy under EOT attack (with and without defenses)
    accuracy_base_teacher = 0
    accuracy_ae = 0
    success_rate = 0

    for img, label in test_data:
        img = img.to(device)
        ae_label = random.randint(0,8)
        ae_label += 1 if ae_label >= label else 0 # ensuring target != label
        y_target = torch.tensor([ae_label], dtype=torch.long, device=device)
        x_adv = eot_attack(model, img, y_target)
        with torch.no_grad():
            pred_clean = model(img).argmax(dim=1)
            pred_adv = model(x_adv).argmax(dim=1)

            if pred_clean == label:
                accuracy_base_teacher += 1
            if pred_adv == label:
                accuracy_ae += 1
            if pred_adv == ae_label:
                success_rate += 1

    # TODO: Save your results and write your short analysis separately
    print("\n--- Part 2: Evaluation Results ---")
    print(f"EOT Clean accuracy: {accuracy_base_teacher/N:.2f}")
    print(f"EOT adversarial accuracy: {accuracy_ae/N:.2f}")
    print(f"EOT attack success rate: {success_rate/N:.2f}")

    # PART 3: Distillation Defense
    # TODO: Train a student model via distillation using the teacher model
    # TODO: Save your model as 'student_VGG.pth'
    student_VGG()
    # TODO: Evaluate student model accuracy on clean and PGD-attacked images
    _, test_loader = load_dataset()
    N = 100
    test_data = []
    counts = {i: 0 for i in range(10)}
    for imgs, labels in test_loader:
        # img in tensor form
        for img, label in zip(imgs, labels):
            label = label.item()
            if counts[label] < N // 10:
                test_data.append((img2tensorVGG(tensor2imgVGG(img), device), label)) # storing img and AE in tensor form
                counts[label] += 1
            if len(test_data) >= N:
                break

    # Load student model
    student = hw3_utils.get_vgg_model().to(device)
    student.load_state_dict(torch.load("models/student_VGG.pth", map_location=device))
    student.eval()

    # Load teacher model for PGD attack
    teacher = VGG('VGG16').to(device)
    teacher.load_state_dict(torch.load("models/vgg16_cifar10_robust.pth", map_location=device))
    teacher.eval()

    accuracy_base = 0
    accuracy_ae = 0
    accuracy_base_teacher = 0


    for img, label in test_data:
        img = img.to(device)
        # Base
        with torch.no_grad():
            pred_clean = student(img).argmax(dim=1)
            if pred_clean == label:
                accuracy_base += 1
            pred_teacher = model(img).argmax(dim=1)
            if pred_teacher == label:
                accuracy_base_teacher += 1

        # PGD attack
        target_label = random.randint(0, 8)
        target_label += 1 if target_label >= label else 0
        adv_img = target_pgd_attack(tensor2imgVGG(img), target_label, teacher, device)
        adv_tensor = img2tensorVGG(adv_img, device)

        # Adversarial prediction
        with torch.no_grad():
            pred_adv = student(adv_tensor).argmax(dim=1)
            if pred_adv == label:
                accuracy_ae += 1


    # TODO: Save or print results and include your writeup separately
    print("\n--- Part 3: Distillation Evaluation ---")
    print(f"Teacher base accuracy: {accuracy_base_teacher / N:.2f}")
    print(f"Student base accuracy: {accuracy_base / N:.2f}")
    print(f"Student PGD adversarial accuracy: {accuracy_ae / N:.2f}")

    

if __name__ == "__main__":
    main()
