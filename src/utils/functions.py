import torch
from torchvision import transforms


def custom_pre_process_function():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform


def custom_loss_function(output, target):
    loss = torch.mean((output - target)**2)
    return loss
