import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.kl import kl_divergence as KL

from torch import Tensor
from math import log, pi

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, nonlinearity):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.input_size = input_size
        self.latent_size = latent_size
        self.nonlinearity = nonlinearity

        # Layers
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))

        for i in range(len(hidden_sizes)-1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        self.fc_mean = nn.Linear(hidden_sizes[-1], latent_size)
        self.fc_logvar = nn.Linear(hidden_sizes[-1], latent_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for layer in self.hidden_layers:
            x = self.nonlinearity(layer(x))
        z_mean = self.fc_mean(x)
        z_logvar = self.fc_logvar(x)
        return z_mean, z_logvar


class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_sizes, output_size, nonlinearity):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.latent_size = latent_size
        self.nonlinearity = nonlinearity
        # layers
        self.hidden_layers.append(nn.Linear(latent_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes)-1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, z):
        for layer in self.hidden_layers:
            z = self.nonlinearity(layer(z))
        x_recon = torch.sigmoid(self.output_layer(z))
        return x_recon


class SuperVAE(nn.Module):
    def __init__(self, input_size, enc_hidden_sizes, 
                 dec_hidden_sizes, latent_size,
                 var=1, 
                 dimension_decrease=0, name='model name',
                 enc_nonlinearity=nn.ReLU(), dec_nonlinearity=nn.ReLU()):
        super().__init__()
        dec_latent = latent_size - dimension_decrease
        self.encoder = Encoder(input_size, enc_hidden_sizes, latent_size, enc_nonlinearity)
        self.decoder = Decoder(dec_latent, dec_hidden_sizes, input_size, dec_nonlinearity)
        self.latent_size = latent_size
        self.name = name
        self.loss = (0,0,0)
        self.var = var # observation noise prediction

    def reparameterize(self, mu, logvar, 
                       ind_bias=False, distribution=None, alpha=0, beta=0):
        var = torch.exp(0.5*logvar)
        z = Normal(mu, var).rsample()
        if ind_bias:
            c = distribution(alpha, beta).rsample()
            return z, c
        return z

    def KL_normal(self, mu, logvar):
        '''
                KL divergence of the given normal distribution from N(0, I)
        '''
        return -0.5 * torch.sum((self.latent_size + torch.sum(logvar - mu.pow(2) - logvar.exp(), dim=1)))
	
    def loss_function(self, x_recon, x, mu, logvar, is_bce):
        n = len(x)

        KLD = self.KL_normal(mu, logvar)

        if is_bce:
            REC = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        else:
            REC = F.mse_loss(x_recon, x.view(-1, 784), reduction='sum')
        REC = (n/2) * log(self.var) + REC / (2*self.var)
        self.loss = REC + KLD, REC, KLD
        return self.loss


class VAE(SuperVAE):
    def __init__(self, input_size, enc_hidden_sizes,
                 dec_hidden_sizes, latent_size, var=1,
                 enc_nonlinearity=nn.ReLU(), dec_nonlinearity=nn.ReLU()):
        super().__init__(input_size, enc_hidden_sizes, dec_hidden_sizes, latent_size, var, name='Standard VAE')
				
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
        
class SMVAE_NORMAL(SuperVAE):
    def __init__(self, input_size, enc_hidden_sizes,
                dec_hidden_sizes, latent_size, var=1,
                enc_nonlinearity=nn.ReLU(), dec_nonlinearity=nn.ReLU()):
        super().__init__(input_size, enc_hidden_sizes, dec_hidden_sizes, 
                         latent_size, var, dimension_decrease=1, name='Normal SMVAE')

    def forward(self, x):
            mu, logvar = self.encoder(x)
            rep = self.reparameterize(mu, logvar)
            z = rep[:, :self.latent_size-1]
            c = rep[:, self.latent_size-1]
            c = c.reshape(-1, 1)
            c = torch.sigmoid(c)
            x_recon = self.decoder(z)
            x_recon = x_recon*c
            return x_recon, mu, logvar    

class SMVAE_BETA(SuperVAE):    
    def __init__(self, input_size, enc_hidden_sizes,
                dec_hidden_sizes, latent_size, var=1,
                enc_nonlinearity=nn.ReLU(), dec_nonlinearity=nn.ReLU()):
        super().__init__(input_size, enc_hidden_sizes, dec_hidden_sizes, 
                         latent_size, var, dimension_decrease=1, name='Beta SMVAE')
    
    def get_params(self, mean, var):
        mu = mean[:,:-1]
        logvar = var[:,:-1]
        
        alpha = torch.exp(mean[:,-1])
        beta = torch.exp(var[:,-1])

        return mu, logvar, alpha, beta

    def forward(self, x):
        mean, var = self.encoder(x)
        mu, logvar, alpha, beta = self.get_params(mean, var)
        z, c = self.reparameterize(mu, logvar, True, Beta, alpha, beta)
        c = c.reshape(-1, 1)
        x_recon = self.decoder(z)
        x_recon = x_recon*c
        return x_recon, mean, var  
    
    def loss_function(self, x_recon, x, mean, var, is_bce):
        prior_alpha, prior_beta = Tensor([1]), Tensor([1])
        mu, logvar, alpha, beta = self.get_params(mean, var)

        n = self.latent_size - 1
        KL_normal = KL(Normal(mu, torch.exp(0.5*logvar)), Normal(torch.zeros(n), torch.ones(n))) # batch_size x latent_size-1
        KL_normal = torch.sum(KL_normal, 1) # sum latents in each sample
        KL_beta = KL(Beta(alpha, beta), Beta(prior_alpha, prior_beta))
        KLD = KL_normal + KL_beta

        if is_bce:
            REC = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        else:
            REC = F.mse_loss(x_recon, x.view(-1, 784), reduction='sum')
        REC = (n/2) * log(self.var) + REC / (2*self.var)
        print(KLD.shape, REC.shape)
        self.loss = REC + KLD, REC, KLD
        return self.loss

class SMVAE_LOGNORMAL(SuperVAE):
    def __init__(self, input_size, enc_hidden_sizes,
                dec_hidden_sizes, latent_size, var=1,
                enc_nonlinearity=nn.ReLU(), dec_nonlinearity=nn.ReLU()):
        super().__init__(input_size, enc_hidden_sizes, dec_hidden_sizes, 
                         latent_size, var, dimension_decrease=1, name='Lognormal SMVAE')

    def forward(self, x):
            mu, logvar = self.encoder(x)
            rep = self.reparameterize(mu, logvar)
            z = rep[:, :self.latent_size-1]
            c = rep[:, self.latent_size-1]
            c = c.reshape(-1, 1)
            c = torch.exp(c)
            x_recon = self.decoder(z)
            x_recon = x_recon*c
            x_recon = torch.clamp(x_recon, max=1)
            return x_recon, mu, logvar

class SMVAE_GAMMA(SuperVAE):
    def __init__(self, input_size, enc_hidden_sizes,
                dec_hidden_sizes, latent_size, alpha=1, var=1,
                enc_nonlinearity=nn.ReLU(), dec_nonlinearity=nn.ReLU()):
        gname = 'Gamma(' + str(alpha) + ') SMVAE'
        super().__init__(input_size, enc_hidden_sizes, dec_hidden_sizes, 
                         latent_size, var, dimension_decrease=1, name=gname)
        self.alpha = alpha

    def get_params(self, mean, var):
        mu = mean[:,:-1]
        logvar = var[:,:-1]

        g_logmean = mean[:,-1]
        g_logvar = var[:,-1]
        alpha = torch.exp(g_logmean)**2 / torch.exp(g_logvar)
        beta = torch.exp(g_logmean) / torch.exp(g_logvar)

        return mu, logvar, alpha, beta

    def forward(self, x):
        mean, var = self.encoder(x)
        mu, logvar, alpha, beta = self.get_params(mean, var)
        z, c = self.reparameterize(mu, logvar, True, Gamma, alpha, beta)
        c = c.reshape(-1, 1)
        x_recon = self.decoder(z)
        x_recon = x_recon*c
        x_recon = torch.clamp(x_recon, max=1)
        return x_recon, mean, var

    def KL_gamma(self, alpha, beta, prior_alpha, prior_beta):
        '''
                KL divergence of the given Gamma distribution from the prior Gamma
        '''
        kl = torch.sum(\
            (alpha-prior_alpha)*torch.digamma(alpha) \
            - (beta-prior_beta)*alpha/beta \
            + alpha*torch.log(beta) \
            + torch.lgamma(prior_alpha) \
            - torch.lgamma(alpha))
        return kl

    def loss_function(self, x_recon, x, mean, var, is_bce):
        prior_alpha, prior_beta = Tensor([self.alpha]), Tensor([1])
        mu, logvar, alpha, beta = self.get_params(mean, var)
			
        KLD = self.KL_normal(mu, logvar) \
            + self.KL_gamma(alpha, beta, prior_alpha, prior_beta)
        #KLD = self.var*KLD

        if is_bce:
            REC = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        else:
            REC = F.mse_loss(x_recon, x.view(-1, 784), reduction='sum')
        self.loss = REC + KLD, REC, KLD
        return self.loss