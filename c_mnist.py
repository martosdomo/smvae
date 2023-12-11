# CONTRAST AUGMENTED MNIST DATASET

import random
#from torchvision import datasets
#from torch.utils.data import DataLoader
from torchvision import transforms

# CONTRAST FUNCTIONS

def fixed_contrast(data, c_list):
    new_data = []
    for img, label in data:
        for c in c_list:
            img_aug = img * c
            new_data.append((img_aug, label))
    return new_data

def random_contrast(data, min_contrast, max_contrast, length,
                    single_contrast_labels={}, single_contrast=1):
    new_data = []
    size = 0

    for img, label in data:
        if label in single_contrast_labels:
            mycontrast = single_contrast
        else:
            mycontrast = random.uniform(min_contrast, max_contrast)
            mycontrast = round(mycontrast, 4)
        img_aug = img * mycontrast
        new_data.append((img_aug, (label, mycontrast)))
        size += 1
        if size >= length:
            return new_data            
    return new_data

# CREATE TEST & TRAIN SETS

def create_trainset(dataset, full_length,
                    min_contrast, max_contrast,
                    single_contrast_labels={}, single_contrast=1):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = dataset(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    trainset = random_contrast(trainset, min_contrast, max_contrast, full_length, single_contrast_labels, single_contrast)
    return trainset

def create_testset(dataset, min_contrast=0, max_contrast=1, 
                   length=10000, return_validation=True):
    transform = transforms.Compose([transforms.ToTensor()])
    testset = dataset(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    testset = random_contrast(testset, min_contrast, max_contrast, length)

    validation = testset[:5000]
    testset = testset[5000:]

    if not return_validation:
            return testset
    return testset, validation
