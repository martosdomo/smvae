from saveload import *
from c_mnist import *
from models import *
from functions import *

import torch
from torchvision.datasets import MNIST, FashionMNIST

def main():

    # random seed
    random_seed = 1
    set_seed(random_seed)

    # hyperparams
    BATCH_SIZE = 32
    LEARNING_RATE = 0.5*1e-3  
    NUM_EPOCHS = 1000

    # obs_noise values
    opt_sigmas = {'[0, 1]': .01, 
                  '[0.8, 1]': .01,
                  '[0, 0.2]': .03}
    #sigmas = [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    
    # dataset
    MNIST.name, FashionMNIST.name = 'mnist', 'fashion'
    DATA = MNIST
    sizes = [60000, 20000, 6000, 2000, 600]
    #sizes = [600]
    contrasts = [[0, 1], [0.8, 1], [0, 0.2]]
    datasets = []
    for size in sizes:
        for contrast in contrasts:
            trainset = create_trainset(DATA, size, contrast[0], contrast[1])
            testset, validation = create_testset(DATA, 0, 1)
            sigma = opt_sigmas[str(contrast)]
            datasets.append([trainset, validation, testset, [size, contrast], [sigma]])
    
    # train
    full_logs = []
    for dataset in datasets:
        #models = get_models(sigmas, VAE, SMVAE_NORMAL, SMVAE_BETA)
        models = get_models(dataset[4], VAE, SMVAE_NORMAL, SMVAE_BETA)
        for model in models:
            # model.name_sizeofdata_mincontrast_maxcontrast_obsnoise_randomseed
            model.name += '_{}_{}_{}_{}_var{}_seed{}'.format(DATA.name, dataset[3][0], dataset[3][1][0], dataset[3][1][1], model.var, random_seed)
            myloss, mycheckpoints, max_validation, logs = train(model, dataset[0], dataset[1], 
                                                          LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS)
            
            print('Max validation: ', max_validation)
            model.load_state_dict(mycheckpoints[max_validation])
            
            full_logs.append(logs)
            save_model(model, model.type+'/'+model.name+'.pth')
            plot(myloss, model.name)
    torch.save(full_logs, '//mnt/smvae/evaluation/logs_{}_seed{}.txt'.format(DATA.name, random_seed))

if __name__ == '__main__':
    main()