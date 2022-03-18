#GGTM architecture
from os import name
import pandas as pd
import numpy as np
import matplotlib as plt
from collections import Counter
from ggtminit import *
from mixmodel import *


dim_latent = 2
latent_shape = [8,8]
nlatent = np.prod(latent_shape)
rbf_grid = [4,4]
num_rbf_centres = np.prod(rbf_grid)
rbf_prior = 0.01
samp_type = 'regular'

#Mapping function
Mapping = {}
Mapping['type'] = []
Mapping['type'].append('rbf')
# Mapping['type'].append('gp')
Mapping['func'] = []
Mapping['func'].append('gaussian')
Mapping['ncentres'] = num_rbf_centres
Mapping['prior'] = rbf_prior

#Data loading
#Continuous data
df_1 = pd.read_csv("cont_train_data.csv")
cont_train_data = df_1.drop(index = 0, columns = ['ID','Label']).astype(float)
cont_data = cont_train_data.to_numpy()
v1 = cont_train_data['v1'].to_numpy()
v2 = cont_train_data['v2'].to_numpy()
v3 = cont_train_data['v3'].to_numpy()
pre = {'mu':[np.mean(v1),np.mean(v2),np.mean(v3)], 'sigma':[1.,1.,1.]}
cdata =  {'mat':cont_data, 'type' : 'continuous', 'nvar' : cont_data.shape[1]}
#mixture model
cmix = {'type':'gmm', 'covar_type':'spherical'}

#Binary data

df_2 = pd.read_csv("bin_train_data.csv")
bin_train_data = df_2.drop(index = 0, columns = ['ID','Label']).astype(float)
bin_data = bin_train_data.to_numpy()
bdata = {'mat':bin_data, 'type':'discrete', 'nvar':bin_data.shape[1]}
print(bin_data.shape[1])
#mixture model
bmix = {'type': 'dmm', 'covar_type': 'spherical','dist_type': 'bernoulli','cat_nvals': 2}

#Categorical data
df_3 = pd.read_csv("cat_train_data.csv")
cat_train_data = df_3.drop(index = 0, columns = ['ID','Label']).astype(float)
cat_data = cat_train_data.to_numpy()
cat_count = len(Counter(df_3['Label']))-1
cadata = {'mat' : cat_data, 'cat_nvals': cat_count, 'type': 'discrete', 'nvar' : cat_data.shape[1]}
#Mixture model
camix = {'type': 'dmm', 'covar_type': 'spherical', 'dist_type': 'categorical',
       'cat_nvals':cadata['cat_nvals']}

# Creationg of Data Array
data_array = np.array([cdata, bdata, cadata])
ndata = data_array.shape[0]
dim_data_array = []
for i in range(0, ndata):
    dim_data_array.append(data_array[i]['nvar'])

#Creationg of  Mixture models array
mix_array = np.array([cmix, bmix, camix])

# Create and initialise GTM model
net = ggtm(dim_latent, nlatent, dim_data_array, Mapping, mix_array)

net = ggtminit(net, data_array, samp_type, latent_shape,rbf_grid)

[net, options] = ggtmem(net, data_array)
# Posterior means
means = ggtmlmean(net, data_array)
plt.plot(means[:, ], means[:, 2], 'k.')



if __name__ == "__main__" :
    plt.plot(means[:, ], means[:, 2], 'k.')
