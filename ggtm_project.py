from typing import Match
import numpy as np
import matplotlib as plt


class ggtm():
    """constructor for class of model GGTM
    parameters:
    DIM_LATENT, the dimension of the latent space.
    NLATENT, the number of data points in the latent space.
    DIM_DATA_ARRAY, the vector consists of dimensions of each data space type.
    MAP_FUNC, the parameters of mapping function and mixture models
    MIX_ARRAY, the mixtuer model of each observation space, where obervation space represents each type of data.
    """

    def __init__(self, dim_latent, nlatent, dim_data_array, mix_array):
        
        self.dim_latent = dim_latent
        self.nlatent = nlatent
        self.dim_data_array = dim_data_array
        self.mapping = Mapping()
        self.mix_array = mix_array
    
    def Obs(self):
        """Find the number of input variable
        """
        ndata_space = np.shape(self.dim_data_array,2)
        obs = []
        for i in range(0,ndata_space):
            obs = self.dim_data_array[i]
        
        
        
        
    




class Mapping():
    """creating a map stucture
     The MAP structure is defined as follows
	    type = 'rbf' or 'gp'
	    func = RBF activation function or GP covariance function
	    prior = scalar representing the inverse variance of the RBF prior distribution
	    ncentres = number of RBF centres

    """
    
    def __init__(self,type, func, rbf_prior, rbf_grid =[]):
        self.type = type
        self.func = func
        self.rbf_prior = rbf_prior
        self.rbf_grid = []
    
    def N_Centres(self):
        ncentres = np.prod(self.rbf_grid)
        return ncentres


    def MappingType(self):
        """
        Determine the mapping type
    
        """
        prior = self.rbf_prior
        
        if  self.type == 'rbf':
            
            obs_map = rbf(ggtm.dim_latent, self.N_Centres(), ggtm.Obs(), self.func, 'linear', prior)
            obs_map_mask = rbfprior(ggtm.dim_latent, self.NCentres(), ggtm.Obs(), self.func)
        
        elif self.type == 'gp':
            
            obs_map = gp(ggtm.Obs(), self.func, prior)
        
        else:
            print('Unknown mapping function.')
    
    def Mixture_Model(self,obs_type):
        """Determine the mixture model"""
        if obs_type == 'continuous':
            obs_mix = gmm(obs, nlatent, mix_array)





        






    







            


        
        
        
        
        
        
        
        
        # mapping = np.array([('rbf', 'gaussian', rbf_prior, num_rbf_centres), ()],
        # dtype = [('type'), ('func'), ('prior'), ('ncentres')])
