import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from functions import *

def create_df_from_dict(results):
    '''
        Creates a detailed dataframe from the elbo results
    '''
    df = pd.DataFrame.from_dict(results, orient='index')
    df = df.reindex(sorted(df.index), axis=0)
    df = df.rename(columns={0:'ELBO', 1:'REC', 2:'neg_KL'})
    df = df.round(2)

    df['type'] = [ind.split('_')[0]+'_'+ind.split('_')[1] for ind in df.index]
    df['data'] = [ind.split('_')[2] for ind in df.index]
    df['size'] = [int(ind.split('_')[3]) for ind in df.index]
    df['contrast'] = [str(ind.split('_')[4:6]).replace("'","") for ind in df.index]
    df['var'] = [float(ind.split('_')[6][3:]) for ind in df.index]
    df['seed'] = [int(ind.split('_')[7][-1:]) for ind in df.index]

    df = df[df['type']!='Beta_SMVAE']

    df_full = df[df['contrast']=="[0, 1]"]
    df_hi = df[df['contrast']=="[0.8, 1]"]
    df_lo = df[df['contrast']=="[0, 0.2]"]
    
    return df, df_full, df_hi, df_lo

def plot_contrast(contrasts, c_means, title, regression=True, c='blue', limit=True):
    '''
        Scatter plot of the contrast and the posterior mean
    '''
    plt.scatter(contrasts, c_means, s=1, color=c)

    # Perform linear regression using PyTorch
    if regression:
        x = contrasts.view(-1, 1)  # Reshape to a column vector
        y = c_means.view(-1, 1)  # Reshape to a column vector
        X = torch.cat([x, torch.ones_like(x)], dim=1)  # Augmented input matrix [x, 1]
        w, _ = torch.lstsq(y, X)
        regression_line = x * w[0] + w[1]
        plt.plot(contrasts, regression_line, label='Regression Line', color='red')
    
    if limit:
        plt.ylim(-0.1, 1.1)
    plt.xlabel('Contrast ground truth')
    plt.ylabel('Posterior mean of c')
    plt.title(title)

    # Display the plot
    plt.show()
    
def plot_elbos(*args):
    '''
        Plots the elbos of the given models for the various data sizes
    '''
    fig, ax = plt.subplots()
    mydf = args[0][0]
    title = mydf['data'][0] + '_' + mydf['contrast'][0] + '_seed' + str(mydf['seed'][0])
    labels = [600, 2000, 6000, 20000, 60000]
    
    for df_and_color in args:
        df, c = df_and_color[0], df_and_color[1]
        values = list(df['ELBO'])
        ax.plot(range(1, len(labels)+1), values, marker='o', linestyle=':', color=c, label=df['type'][0])

    plt.grid(axis='y')

    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, rotation=45)  # You can adjust the rotation angle as needed

    '''for i, value in enumerate(values):
        ax.text(i + 1, value, str(value), fontsize=10, ha='center', va='bottom')
    for i, value in enumerate(values2):
        ax.text(i + 1, value, str(value), fontsize=10, ha='center', va='bottom')'''
    
    ax.set_xlabel('Training set size')
    ax.set_ylabel('ELBO')
    ax.set_title(title)
    ax.legend()
    plt.show()
    
def df_subset(df_contrast, myseed, mytype, mydata):
    '''
        Returns a given subset of the evaluation dataframe
    '''
    mydf = df_contrast[(df_contrast.seed==myseed)&(df_contrast.type==mytype)&(df_contrast.data==mydata)]
    mydf = mydf.sort_values('size', ascending=True)
    return mydf

def corr_barplot(means, variances, contrasts, model, testset, color):
    
    correlation = np.corrcoef(means.T, contrasts.reshape(1,len(testset)))[:-1,-1]
    x_labels = ['$z_{%d}$' % (i+1) for i in range(len(correlation))]
    
    colors = [color]*len(correlation)
    if 'Standard' not in model.name:
        colors[-1] = 'red'
        x_labels[-1] = '$c$'
        
    plt.bar([i for i in range(len(correlation))], correlation, color=colors, tick_label=x_labels)
  
    plt.grid(axis='y')
    plt.ylim(-1.1, 1.1)
    plt.title('Correlation of posterior means and contrast\n'+model.name)

    plt.show()