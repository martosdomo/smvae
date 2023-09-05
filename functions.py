import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

def train(model, trainset, is_bce, learning_rate, batch_size, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainloader = DataLoader(trainset, batch_size, shuffle=True)
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        running_reconstr = 0.0
        running_regul = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, _ = data
            optimizer.zero_grad()
            x_recon, mu, logvar = model(inputs)
            #print(mu, logvar) # WORKING
            loss, reconstr, regul = model.loss_function(x_recon, inputs, mu, logvar, is_bce)
            #print('loss: ', loss, reconstr, regul)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_reconstr += reconstr
            running_regul += regul
        epoch_loss = running_loss/len(trainset)
        epoch_reconstr = running_reconstr/len(trainset)
        epoch_regul = running_regul/len(trainset)
        losses.append([epoch_loss, epoch_reconstr.detach().numpy(), epoch_regul.detach().numpy()])

        print('Epoch [%d/%d], Loss: %.3f, Reconstruction: %.3f, Regularization: %.3f'
              % (epoch+1, epochs, epoch_loss, epoch_reconstr, epoch_regul))
    return losses

def plot(data, str='Title'):
    plt.figure()
    plt.plot(data)
    plt.title(str)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

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

def barplot(data, ylabel='ylabel', title='Title'):
    z1_group = data[:,0].detach().numpy().flatten()
    z2_group = data[:,1].detach().numpy().flatten()
    c_group = data[:,2].detach().numpy().flatten()

    axis_labels = ['c = ' + str(contrast_values[i]) for i in range(4)]

    positions = np.array([i for i in range(0, 13, 4)])
    z1_pos = positions
    z2_pos = positions + 1
    c_pos = positions + 2

    fig, ax = plt.subplots()

    plt.bar(z1_pos, z1_group, label='$z_1$', color='forestgreen')
    plt.bar(z2_pos, z2_group, label='$z_2$', color='limegreen')
    plt.bar(c_pos, c_group, label='$c$', color='r')

    # Assigning group labels to the x-axis
    plt.xticks([1+4*d for d in range(4)], axis_labels)

    # Adding legend
    plt.legend()

    plt.grid(axis='y')
    ax.set_axisbelow(True)

    # Display the bar chart
    plt.xlabel('Kontraszt értékek')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def get_avereges(model, LATENT_SIZE, testset):

    # PARAMS
    n = 10000
    contrasts = len(contrast_values)
    post_sum = torch.zeros(4, 2, LATENT_SIZE)
    dif_chars = n // contrasts

    # REGRESSION DATA
    cs = torch.zeros(n)
    means = torch.zeros(n,LATENT_SIZE)
    vars = torch.zeros(n,LATENT_SIZE)
    #means = vars = torch.zeros(n, LATENT_SIZE)

    # GET POSTERIOR AVEREGES AND REGRESSION DATA

    for i in range(0, n, contrasts):
        for c in range(contrasts):
            # mean
            # post_sum[c][0]: c a megfelelő kontraszt, 0 a mean értékek
            # encoder[0][0]: 0 a mean érték, második 0 a grad info elhagyása
            mean = model.encoder(testset[i+c][0])[0][0]
            #mean[2] = torch.exp(mean[2])

            # var
            logvar = model.encoder(testset[i+c][0])[1][0]
            var = torch.exp(0.5*logvar)
            #var[2] = torch.exp(var[2])

            # for lognormal
            """mean_logn = torch.exp(mean[2] + var[2]**2/2)
            var_logn = torch.sqrt(torch.exp(2*mean[2] + 2*var[2]**2) - torch.exp(2*mean[2] + var[2]**2))
            mean[2] = mean_logn
            var[2] = var_logn"""

            # for gamma
            mean[2] = torch.exp(mean[2])

            post_sum[c][0] += mean
            post_sum[c][1] += var

            # REGRESSION DATA
            cs[i+c] = contrast_values[c]
            means[i+c] = mean#[2]
            vars[i+c] = var#[2]

    post_mean = post_sum / dif_chars

    return post_mean, cs, means, vars

def ELBO(model, is_bce, testset):
    running_loss = 0.0
    running_reconstr = 0.0
    running_regul = 0.0
    for input in testset:
        x_recon, mu, logvar = model(input[0])
        loss, reconstr, regul = model.loss_function(x_recon, input[0], mu, logvar, is_bce)
        running_loss += loss
        running_reconstr += reconstr
        running_regul += regul
    epoch_loss = running_loss / len(testset)
    epoch_reconstr = running_reconstr / len(testset)
    epoch_regul = running_regul / len(testset)

    return epoch_loss, epoch_reconstr, epoch_regul