from saveload import *
from c_mnist import *
from models import *
from functions import *

import torch
from torchvision.datasets import MNIST, FashionMNIST

def main():
    
    # load models
    models_standard = load_folder(VAE, 'standard')
    models_normal = load_folder(SMVAE_NORMAL, 'normal')
    models_beta = load_folder(SMVAE_BETA, 'beta')
    models = models_standard + models_normal + models_beta

    for model in models:
        print(model)

if __name__ == '__main__':
    main()