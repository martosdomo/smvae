from saveload import *
from c_mnist import *
from models import *
from functions import *
import sys

#import torch
from torchvision.datasets import MNIST, FashionMNIST

def main():
    # random seed
    random_seed = 1
    set_seed(random_seed)

    # load dataset
    MNIST.name, FashionMNIST.name = 'mnist', 'fashion'
    DATA = FashionMNIST
    testset = create_testset(DATA, return_validation=False)

    # load models
    models_standard = load_folder(VAE, 'standard', random_seed)
    models_normal = load_folder(SMVAE_NORMAL, 'normal', random_seed)
    models_beta = load_folder(SMVAE_BETA, 'beta', random_seed)
    models = models_standard + models_normal + models_beta

    # evaluate
    evaluations = {}
    for model in models:
        elbo, reconstr, regul = ELBO(model, testset)
        evaluations[model.name] = [elbo.item(), reconstr.item(), regul.item()]
        print(model.name, evaluations[model.name])

    save_file(evaluations, 'eval_{}_seed{}.txt'.format(DATA.name, random_seed))

if __name__ == '__main__':
    main()