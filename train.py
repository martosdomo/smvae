from saveload import *
from c_mnist import *
from models import *
from functions import *

import torch
from torchvision.datasets import MNIST, FashionMNIST

def main():

    # random seed
    set_seed(0)

    # hyperparams
    BATCH_SIZE = 32
    LEARNING_RATE = 0.5*1e-3  
    NUM_EPOCHS = 10

    # dataset
    #sizes = [60000, 20000, 6000, 2000, 600]
    sizes = [600]
    contrasts = [[0, 1], [0, 0.2], [0.8, 1]]
    datasets = []
    for size in sizes:
        for contrast in contrasts:
            trainset = create_trainset(MNIST, size, contrast[0], contrast[1])
            testset, validation = create_testset(MNIST, contrast[0], contrast[1])
            datasets.append([trainset, validation, testset, [size, contrast]])

    # model
    #sigmas = [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    '''sigmas = [0.01]
    models = []
    for sigma in sigmas:
        models.append(VAE(var=sigma))
        models.append(SMVAE_NORMAL(var=sigma))
        models.append(SMVAE_BETA(var=sigma))'''
    
    # train
    for dataset in datasets:
        models = [VAE(), SMVAE_NORMAL(), SMVAE_BETA()]
        for model in models:
            model.name+='_%d_%.1f_%.1f' % (dataset[3][0], dataset[3][1][0], dataset[3][1][1])
            myloss, mycheckpoints, max_validation = train(model, dataset[0], dataset[1], 
                                                          LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS)
            
            print('Max validation: ', max_validation)
            model.load_state_dict(mycheckpoints[max_validation])

            save_model(model, model.type+'/'+model.name+'.pth')
            plot(myloss, model.name)
            

if __name__ == '__main__':
    main()