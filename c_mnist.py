# CONTRAST AUGMENTED MNIST DATASET

import random
from torchvision import datasets
from torch.utils.data import DataLoader

# CONTRAST FUNCTIONS

def fixed_contrast(data, c_list):
    new_data = []
    for img, label in data:
        for c in c_list:
            img_aug = img * c
            new_data.append((img_aug, label))
    return new_data

def random_contrast(data, min_contrast, max_contrast,
                    digit_instances, length):
    new_data = []
    size = 0

    for img, label in data:
        for i in range(digit_instances):
            c = random.uniform(min_contrast, max_contrast)
            c = round(c, 4)
            img_aug = img * c
            new_data.append((img_aug, (label, c)))
            size += 1
            if size >= length:
                return new_data
    return new_data

# CREATE TEST & TRAIN SETS

def create_trainset(digit_instances, length,
                    min_contrast, max_contrast):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    trainset = random_contrast(trainset, min_contrast, max_contrast,
                               digit_instances, length)
    #trainset = trainset[:length]

    return trainset

def create_testset(digit_instances, min_contrast=0, max_contrast=1):
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    testset = random_contrast(testset, min_contrast, max_contrast, digit_instances, 200000)

    return testset
