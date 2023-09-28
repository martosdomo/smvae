import torch
import os
#from google.colab import files

def save_model(model, savename,
               input_size=784, enc_hidden_sizes=[256, 32], dec_hidden_sizes=[32, 256]):
    
    checkpoint = {'input_size': input_size,
                  'enc_hidden_sizes': enc_hidden_sizes,
                  'dec_hidden_sizes': dec_hidden_sizes,
                  'latent_size': model.latent_size,
                  'obs_noise': model.var,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, 'C:/Users/marto/Documents/smvae/'+savename)
    #files.download(savename)

def load_model(CLASS, filepath):
    checkpoint = torch.load(filepath)
    model = CLASS(checkpoint['input_size'],
                  checkpoint['enc_hidden_sizes'],
                  checkpoint['dec_hidden_sizes'],
                  checkpoint['latent_size'],
                  checkpoint['obs_noise'])
    
    model.load_state_dict(checkpoint['state_dict'])

    return model

def load_folder(CLASS, model_type):
    models = []
    directory = 'C:/Users/marto/Documents/smvae/'+model_type
    print(directory)
    for filename in os.listdir(directory):
        models.append(load_model(CLASS, directory+'/'+filename))
    return models