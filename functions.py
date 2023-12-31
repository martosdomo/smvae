import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import random
#from smvae.saveload import *

contrast_values = [0.2, 0.5, 0.8, 1]

def train(model, trainset, validation, learning_rate, batch_size, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainloader = DataLoader(trainset, batch_size, shuffle=True)
    losses, checkpoints, logs = [], [], []
    
    #early stopping 
    training_steps = 10**7
    max_validation = 0
    patience = 20
    eps=0
    
    print(model.name, '| sigma =', model.var)
    logs.append(model.name)

    for epoch in range(epochs):
        running_loss = 0.0
        running_reconstr = 0.0
        running_regul = 0.0
        for i, data in enumerate(trainloader):
            inputs, _ = data
            optimizer.zero_grad()
            x_recon, mu, var = model(inputs)
            #print(x_recon, mu, var) # WORKING
            loss, reconstr, regul = model.loss_function(x_recon, inputs, mu, var, batch_size)
            #print('loss: ', loss, reconstr, regul)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() # item: tensor -> number
            running_reconstr += reconstr.item() # plot miatt
            running_regul += regul.item()
        epoch_loss = -1*running_loss / len(trainset)
        epoch_reconstr = -1*running_reconstr / len(trainset)
        epoch_regul = -1*running_regul / len(trainset)
        
        epoch_valid = ELBO(model, validation)[0].detach().numpy()

        losses.append([epoch_loss, epoch_valid])
        checkpoints.append(model.state_dict())
        
        if (epoch+1)*len(trainset) > training_steps:
            return losses, checkpoints, max_validation, logs
        
        # return if not improved {eps} in the last {patience} epochs
        prev_max = losses[max_validation][1]
        if epoch_valid > prev_max + eps*abs(prev_max):
            max_validation = epoch
        if max_validation < (epoch-patience):
            return losses, checkpoints, max_validation, logs
        
        print_log = 'Epoch %d, Training ELBO: %.3f, Reconstruction: %.3f, Regularization: %.3f || Validation ELBO: %.3f' % (epoch+1, epoch_loss, epoch_reconstr, epoch_regul, epoch_valid)
        print(print_log)
        logs.append(print_log)
        
    return losses, checkpoints, max_validation, logs

def plot(data, str='Title'):
    plt.figure()
    plt.plot(data)
    plt.title(str)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training ELBO', 'Validation ELBO'])

def grid(model, coordinates, rows=20, cols=20, title='Title'):
    #x = y = np.linspace(-range, range, resolution)
    #grid = torch.tensor([[i, j, k] for i in x for j in y])
    grid = torch.tensor(coordinates)
    grid = grid.to(torch.float32)
    samples = model.decoder(grid)
    samples = samples.view(rows*cols, 1, 28, 28)
    grid = make_grid(samples, nrow=cols, normalize=True)

    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.show()

def compare(model, testset, n, k=0, title='Title'):
    og = [testset[i+k][0].reshape(28,28) for i in range(n)]
    rec = [torch.clamp(model(testset[i+k][0])[0][0],max=1).detach().numpy().reshape(28,28) for i in range(n)]

    fig, ax = plt.subplots(2, n, figsize=(n,3))
    fig.suptitle(title)
    ax[0, 0].set_title('Eredeti')
    ax[1, 0].set_title('Rekonstruált')
    for i in range(n):
      ax[0, i].imshow(og[i], cmap='gray', vmax=1)
      ax[1, i].imshow(rec[i], cmap='gray', vmax=1)

    plt.show()

def get_averages(model, testset):

    means = torch.zeros(len(testset), model.latent_size)
    vars = torch.zeros(len(testset), model.latent_size)
    contrasts = torch.zeros(len(testset))
    
    for idx, im in enumerate(testset):
        # encoder[0/1][0]: [0/1] - mean/var. második 0 a grad info elhagyása
        mean = model.encoder(im[0])[0][0]
        logvar = model.encoder(im[0])[1][0] 

        # im[1]: tuple(digit, contrast)
        contrast = im[1][1]

        mean, var = model.get_params(mean, logvar) 
        
        means[idx] = mean
        vars[idx] = var
        contrasts[idx] = contrast
    
    return means.detach(), vars.detach(), contrasts.detach()

def get_contr_averages(model, testset):
    
    means = torch.zeros(len(testset), model.latent_size+1)
    vars = torch.zeros(len(testset), model.latent_size)
    contrasts = torch.zeros(len(testset))
    
    for idx, im in enumerate(testset):
        x = im[0]
        x = x.view(-1, model.input_size)
        c = model.contrast_inference(x)
        x = x/c
        # encoder[0/1][0]: [0/1] - mean/var. második 0 a grad info elhagyása
        latent_pars = model.encoder(x)
        mean = latent_pars[0][0]
        logvar = latent_pars[1][0]

        # im[1]: tuple(digit, contrast)
        contrast = im[1][1]

        mean, var = model.get_params(mean, logvar)
        mean = torch.cat((mean, c.view(-1)))
        
        means[idx] = mean
        vars[idx] = var
        contrasts[idx] = contrast
    
    return means.detach(), vars.detach(), contrasts.detach()

def ELBO(model, testset, batch_size=1): #testset batch_size = 1
    running_loss = 0.0
    running_reconstr = 0.0
    running_regul = 0.0
    for input in testset:
        x_recon, mu, var = model(input[0])
        loss, reconstr, regul = model.loss_function(x_recon, input[0], mu, var, batch_size)
        running_loss += loss
        running_reconstr += reconstr
        running_regul += regul
    epoch_loss = -1*running_loss / len(testset)
    epoch_reconstr = -1*running_reconstr / len(testset)
    epoch_regul = -1*running_regul / len(testset)

    return epoch_loss, epoch_reconstr, epoch_regul

    #return 'ELBO: %.3f, Reconstruction: %.3f, Regularization: %.3f' % (epoch_loss, epoch_reconstr, epoch_regul)

def set_seed(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return None

def get_models(sigmas, *args, is_deep=False):
    models = []
    for sigma in sigmas:
        for model in args:
            if is_deep:
                models.append(model(var=sigma, enc_hidden_sizes=[256,100,32]))
            else:
                models.append(model(var=sigma))
    return models

def label_selective(data, category):
    return [data[i] for i in range(len(data)) if data[i][1][0]==category]