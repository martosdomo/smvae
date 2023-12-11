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
        #x_recon = self.output_layer(z)
        return x_recon

class ContrastInference(nn.Module):
    def __init__(self, input_size=784, output_size=1, nonlinearity=nn.Sigmoid()):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.nonlinearity = nonlinearity
        self.input_size = input_size
        
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc(x)
        x = self.nonlinearity(x)
        return x
    
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
        self.input_size = input_size
        self.latent_size = latent_size
        self.name = name
        self.loss = (0,0,0)
        self.var = var # observation noise hyperparameter, sigma**2

    def reparameterize(self, mu, var, 
                       ind_bias=False, distribution=None, alpha=0, beta=0):
        z = Normal(mu, var).rsample()
        if ind_bias:
            c = distribution(alpha, beta).rsample()
            return z, c
        return z

    def KL_normal(self, mu, var): # kl div of N(mu, var) from N(0, I)
        p = Normal(mu, var)
        q = Normal(torch.zeros_like(mu), torch.ones_like(var))
        return torch.sum(KL(p,q))

    def loss_function(self, x_recon, x, mu, var, BATCH_SIZE):
        n = self.input_size

        KLD = self.KL_divergence(mu, var)
        MSE = F.mse_loss(x_recon, x.view(-1, n), reduction='sum')
        # REC: sum{i=1, batch_size} (MSE_i)
        # sum_losses = sum{i=1,batch_size} (logvar + MSE_i/var) =
        # = batch_size*logvar + 1/var * sum(MSE_i)
        #print('BCE : ' + str(REC) + ' KL : ' + str(KLD))
        REC = BATCH_SIZE * (n/2) * log(2*pi*self.var) + MSE / (2*self.var)
        #print('sigma után: ' + str(REC))

        self.loss = REC + KLD, REC, KLD
        return self.loss


class VAE(SuperVAE):
    def __init__(self, input_size=784, enc_hidden_sizes=[256,32],
                dec_hidden_sizes=[32, 256], latent_size=10, var=0.01,
                enc_nonlinearity=nn.ReLU(), dec_nonlinearity=nn.ReLU(),
                myname='Standard_VAE', mytype='standard'):
        super().__init__(input_size, enc_hidden_sizes, dec_hidden_sizes, latent_size, var, name=myname)
        self.type = mytype

    def KL_divergence(self, mu, var):
        return self.KL_normal(mu, var)
    
    def get_params(self, mean, logvar):
        var = torch.exp(0.5*logvar)
        return mean, var

    def forward(self, x):
        mu, logvar = self.encoder(x)
        mu, var = self.get_params(mu, logvar)
        z = self.reparameterize(mu, var)
        x_recon = self.decoder(z)
        #x_recon = torch.sigmoid(x_recon)
        return x_recon, mu, var
        
class VAE_CONTRAST_INFERENCE(VAE):
    def __init__(self, input_size=784, enc_hidden_sizes=[256,32],
                dec_hidden_sizes=[32, 256], latent_size=10, var=0.01,
                enc_nonlinearity=nn.ReLU(), dec_nonlinearity=nn.ReLU()):
        self.type = 'contrast_inference'
        super().__init__(input_size, enc_hidden_sizes, dec_hidden_sizes, latent_size, var, 
                         myname='ContrastInference_VAE', mytype=self.type)
        self.contrast_inference = ContrastInference()

    def get_params(self, mean, logvar):
        var = torch.exp(0.5*logvar)
        return mean, var        
        
    def forward(self, x):
        x = x.view(-1, self.input_size)
        c = self.contrast_inference(x)
        x = x/c
        mu, logvar = self.encoder(x)
        mu, var = self.get_params(mu, logvar)
        z = self.reparameterize(mu, var)
        x_recon = self.decoder(z)
        x_recon = x_recon*c
        return x_recon, mu, var
        
class SMVAE_NORMAL(SuperVAE):
    def __init__(self, input_size=784, enc_hidden_sizes=[256,32],
                dec_hidden_sizes=[32, 256], latent_size=10, var=0.01,
                enc_nonlinearity=nn.ReLU(), dec_nonlinearity=nn.ReLU()):
        super().__init__(input_size, enc_hidden_sizes, dec_hidden_sizes, 
                         latent_size, var, dimension_decrease=1, name='Normal_SMVAE')
        self.type = 'normal'

    def KL_divergence(self, mu, var):
        z_mu, c_mu = mu[:,:-1], mu[:,-1]
        z_var, c_var = var[:,:-1], var[:,-1]
        
        kl_standard = self.KL_normal(z_mu, z_var)
        
        p = Normal(c_mu, c_var)
        q = Normal(torch.zeros_like(c_mu)-4, torch.ones_like(c_var))
        kl_contrasts = torch.sum(KL(p,q))
        
        return kl_standard + kl_contrasts
    
    def get_params(self, mean, logvar):
        var = torch.exp(0.5*logvar)
        return mean, var

    def forward(self, x):
        mu, logvar = self.encoder(x)
        mu, var = self.get_params(mu, logvar)
        rep = self.reparameterize(mu, var)
        z = rep[:, :self.latent_size-1]
        c = rep[:, self.latent_size-1]
        c = c.reshape(-1, 1)
        c = torch.sigmoid(c)
        x_recon = self.decoder(z)
        x_recon = x_recon*c
        return x_recon, mu, var    
    
class SMVAE_BETA(SuperVAE):    
    def __init__(self, input_size=784, enc_hidden_sizes=[256,32],
                dec_hidden_sizes=[32, 256], latent_size=10, var=0.01,
                enc_nonlinearity=nn.ReLU(), dec_nonlinearity=nn.ReLU()):
        super().__init__(input_size, enc_hidden_sizes, dec_hidden_sizes, 
                         latent_size, var, dimension_decrease=1, name='Beta_SMVAE')
        self.type = 'beta'

    def get_params(self, mean, logvar):
        '''
            Get mean and variance of the Normal and Beta dimensions in each dimension
        '''
        mu = mean[:,:-1]
        var = logvar[:,:-1]
        var = torch.exp(0.5*var)

        beta_mean = torch.sigmoid(mean[:,-1])
        beta_sum = torch.exp(logvar[:,-1])

        alpha = beta_mean*beta_sum
        beta = beta_sum - alpha

        beta_var = (alpha*beta) / ((alpha+beta+1) * (alpha+beta)**2)

        return mu, var, alpha, beta        

    def get_params_forward(self, mean, logvar):
        '''
            Get parameters for the Normal and Beta distributions
        '''
        mu = mean[:,:-1]
        var = logvar[:,:-1]
        var = torch.exp(0.5*var)

        beta_mean = torch.sigmoid(mean[:,-1])
        beta_sum = torch.exp(logvar[:,-1])

        alpha = beta_mean*beta_sum
        beta = beta_sum - alpha

        return mu, var, alpha, beta

    def forward(self, x):
        mean, logvar = self.encoder(x)
        mu, var, alpha, beta = self.get_params_forward(mean, logvar)
        z, c = self.reparameterize(mu, var, True, Beta, alpha, beta)
        c = c.reshape(-1, 1) # transpose
        x_recon = self.decoder(z)
        x_recon = x_recon*c

        par1 = (mu, alpha)
        par2 = (var, beta)
        return x_recon, par1, par2

    def KL_divergence(self, par1, par2):
        mu, alpha = par1[0], par1[1]
        var, beta = par2[0], par2[1]

        kl_normal = self.KL_normal(mu, var)

        p_beta = Beta(alpha, beta)
        q_beta = Beta(1, 1)
        kl_beta = torch.sum(KL(p_beta, q_beta))

        return kl_normal + kl_beta

class SMVAE_LOGNORMAL(SuperVAE):
    def __init__(self, input_size=784, enc_hidden_sizes=[256,32],
                dec_hidden_sizes=[32, 256], latent_size=10, var=1,
                enc_nonlinearity=nn.ReLU(), dec_nonlinearity=nn.ReLU()):
        super().__init__(input_size, enc_hidden_sizes, dec_hidden_sizes, 
                         latent_size, var, dimension_decrease=1, name='Lognormal_SMVAE')

    def KL_divergence(self, mu, var):
        return self.KL_normal(mu, var)

    def forward(self, x):
            mu, logvar = self.encoder(x)
            var = torch.exp(0.5*logvar)
            rep = self.reparameterize(mu, var)

            # get params
            z = rep[:, :self.latent_size-1]
            c = rep[:, self.latent_size-1]
            c = c.reshape(-1, 1)
            c = torch.exp(c)

            x_recon = self.decoder(z)
            x_recon = x_recon*c
            x_recon = torch.clamp(x_recon, max=1)
            return x_recon, mu, var

class SMVAE_GAMMA(SuperVAE):
    def __init__(self, input_size=784, enc_hidden_sizes=[256,32],
                dec_hidden_sizes=[32,256], latent_size=10, alpha_prior=1, var=0.01,
                enc_nonlinearity=nn.ReLU(), dec_nonlinearity=nn.ReLU()):
        gname = 'Gamma' + str(alpha_prior) + '_SMVAE'
        super().__init__(input_size, enc_hidden_sizes, dec_hidden_sizes, 
                         latent_size, var, dimension_decrease=1, name=gname)
        self.alpha_prior = alpha_prior
        self.type = 'gamma'

    def get_params(self, mean, logvar):
        mu = mean[:,:-1]
        var = logvar[:,:-1]
        var = torch.exp(0.5*var)

        g_logmean = mean[:,-1]
        g_logvar = logvar[:,-1]
        alpha = torch.exp(g_logmean)**2 / torch.exp(g_logvar)
        beta = torch.exp(g_logmean) / torch.exp(g_logvar)

        return mu, var, alpha, beta

    def forward(self, x):
        mean, logvar = self.encoder(x)
        mu, var, alpha, beta = self.get_params(mean, logvar)
        z, c = self.reparameterize(mu, var, True, Gamma, alpha, beta)
        c = c.reshape(-1, 1)
        x_recon = self.decoder(z)
        x_recon = x_recon*c
        x_recon = torch.clamp(x_recon, max=1)
        
        par1 = (mu, alpha)
        par2 = (var, beta)
        return x_recon, par1, par2

    def KL_gamma(self, alpha, beta, prior_alpha, prior_beta):
        '''
            KL divergence of the given Gamma distribution from the prior Gamma
        '''
        prior_alpha = torch.Tensor([prior_alpha])
        kl = torch.sum(\
            (alpha-prior_alpha)*torch.digamma(alpha) \
            - (beta-prior_beta)*alpha/beta \
            + alpha*torch.log(beta) \
            + torch.lgamma(prior_alpha) \
            - torch.lgamma(alpha))
        return kl
    
    def KL_divergence(self, par1, par2):
        mu, alpha = par1[0], par1[1]
        var, beta = par2[0], par2[1]
        
        kl_normal = self.KL_normal(mu, var)
        
        '''p_gamma = Gamma(alpha, beta)
        q_gamma = Gamma(self.alpha_prior, 1)
        kl_gamma = torch.sum(KL(p_gamma, q_gamma))''' # appaerantly not working correctly due to some pytorch bug
        # so we use the KL implemented by hand    
        kl_gamma = self.KL_gamma(alpha, beta, self.alpha_prior, 1)
            
        return kl_normal + kl_gamma