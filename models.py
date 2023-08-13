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
                 dec_hidden_sizes, latent_size, name='model name',
                 enc_nonlinearity=nn.ReLU(), dec_nonlinearity=nn.ReLU()):
        super().__init__()
        self.encoder = Encoder(input_size, enc_hidden_sizes, latent_size, enc_nonlinearity)
        self.decoder = Decoder(latent_size, dec_hidden_sizes, input_size, dec_nonlinearity)
        self.latent_size = latent_size
        self.name = name
        self.loss = (0,0,0)

		def reparameterize(self, mu, logvar, 
											 alpha=0, beta=0, is_gamma=False):
				var = torch.exp(0.5*logvar)
				z = Normal(mean, var).rsample()
				if is_gamma:
						c = Gamma(alpha, beta).rsample()
						return z, c
				return z

		def KL_normal(self, mu, logvar):
				'''
						KL divergence of the given normal distribution from N(0, I)
				'''
				return -0.5 * torch.sum((self.latent_size + torch.sum(logvar - mu.pow(2) - logvar.exp(), dim=1)))
	
    def loss_function(self, x_recon, x, mu, logvar, is_bce):
        KLD = self.KL_normal(mu, logvar)
        if is_bce:
            REC = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        else:
            REC = F.mse_loss(x_recon, x.view(-1, 784), reduction='sum')
        self.loss = REC + KLD, REC, KLD
        return self.loss


class VAE(SuperVAE):
    def __init__(self, input_size, enc_hidden_sizes,
                 dec_hidden_sizes, latent_size,
                 enc_nonlinearity=nn.ReLU(), dec_nonlinearity=nn.ReLU()):
        super().__init__(input_size, enc_hidden_sizes, dec_hidden_sizes, latent_size, 'Standard VAE')
				
    def forward(self, x):
        mu, logvar = super().encoder(x)
				z = super().reparameterize(mu, logvar)
				x_recon = super().decoder(z)
				return x_recon, mu, logvar
        

class SMVAE_LOGNORMAL(SuperVAE):
		def __init__(self, input_size, enc_hidden_sizes,
                 dec_hidden_sizes, latent_size,
                 enc_nonlinearity=nn.ReLU(), dec_nonlinearity=nn.ReLU()):
				super().__init__(input_size, enc_hidden_sizes, dec_hidden_sizes, latent_size, 'Lognormal SMVAE')

		def forward(self, x):
				mu, logvar = super().encoder(x)
				rep = super().reparameterize(mu, logvar)
				z = rep[:, :super().latent_size-1]
				c = rep[:, super().latent_size-1]
				c = c.reshape(-1, 1)
				c = torch.exp(c)
				#c = torch.sigmoid(c)
				x_recon = super().decoder(z)
				x_recon = x_recon*c
				x_recon = torch.clamp(x_recon, max=1)
				return x_recon, mu, logvar


class SMVAE_GAMMA(SuperVAE):
		def __init__(self, input_size, enc_hidden_sizes,
							 dec_hidden_sizes, latent_size, alpha=1,
							 enc_nonlinearity=nn.ReLU(), dec_nonlinearity=nn.ReLU()):
				name = 'Gamma(' + str(alpha) + ') SMVAE'
				super().__init__(input_size, enc_hidden_sizes, dec_hidden_sizes, latent_size, name)

		def get_params(self, mean, var):
				mu = mean[:,:-1]
				logvar = var[:,:-1]

				g_logmean = mean[:,-1]
				g_logvar = var[:,-1]
        alpha = torch.exp(g_logmean)**2 / torch.exp(g_logvar)
        beta = torch.exp(g_logmean) / torch.exp(g_logvar)

				return mu, logvar, alpha, beta

		def forward(self, x):
				mean, var = super().encoder(x)
				mu, logvar, alpha, beta = self.get_params(mean, var)
				z, c = super().reparameterize(mu, logvar, alpha, beta, is_gamma=True)
				c = c.reshape(-1, 1)
				x_recon = super().decoder(z)
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
        prior_alpha, prior_beta = Tensor([self.a]), Tensor([1])
        mu, logvar, alpha, beta = self.get_params(mean, var)
			
				KLD = super().KL_normal(mu, logvar) \
							 + self.KL_gamma(alpha, beta, prior_alpha, prior_beta)

        if is_bce:
            REC = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        else:
            REC = F.mse_loss(x_recon, x.view(-1, 784), reduction='sum')
        super().loss = REC + KLD, REC, KLD
        return super().loss
