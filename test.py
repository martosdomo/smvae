from saveload import *
from c_mnist import *
from models import *
from functions import *
import sys

#import torch
from torchvision.datasets import MNIST, FashionMNIST

def main():
    # random seed
    random_seed = 0
    set_seed(random_seed)

    # load trainset
    testset = create_testset(MNIST, return_validation=False)

    # load models
    models_standard = load_folder(VAE, 'standard', random_seed)
    models_normal = load_folder(SMVAE_NORMAL, 'normal', random_seed)
    models_beta = load_folder(SMVAE_BETA, 'beta', random_seed)
    models = models_standard + models_normal + models_beta

    # evaluate
    evaluations = {}
    for model in models:
        elbo, reconstr, regul = ELBO(model, testset)
        evaluations[model.name] = [elbo, reconstr, regul]

    save_file(evaluations, 'evaluations_'+str(random_seed)+'.txt')

if __name__ == '__main__':
    main()