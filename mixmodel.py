import numpy as np
import rbfnet as rn
from utilities import *


def gmm(dim, ncentres, covar_type):
    """
    Description
	 MIX = GMM(DIM, NCENTRES, COVARTYPE) takes the dimension of the space
	DIM, the number of centres in the mixture model and the type of the
	mixture model, and returns a data structure MIX. The mixture model
	type defines the covariance structure of each component  Gaussian:
	  'spherical' = single variance parameter for each component: stored as a vector
	  'diag' = diagonal matrix for each component: stored as rows of a matrix
	  'full' = full matrix for each component: stored as 3d array
	  'ppca' = probabilistic PCA: stored as principal components (in a 3d array
	    and associated variances and off-subspace noise
	 MIX = GMM(DIM, NCENTRES, COVARTYPE, PPCA_DIM) also sets the
	dimension of the PPCA sub-spaces: the default value is one.

	The priors are initialised to equal values summing to one, and the
	covariances are all the identity matrix (or equivalent).  The centres
	are initialised randomly from a zero mean unit variance Gaussian.
	This makes use of the MATLAB function RANDN and so the seed for the
	random weight initialisation can be set using RANDN('STATE', S) where
	S is the state value.

	The fields in MIX are
	  
	  type = 'gmm'
	  nin = the dimension of the space
	  ncentres = number of mixture components
	  covartype = string for type of variance model
	  priors = mixing coefficients
	  centres = means of Gaussians: stored as rows of a matrix
	  covars = covariances of Gaussians
	 The additional fields for mixtures of PPCA are
	  U = principal component subspaces
	  lambda = in-space covariances: stored as rows of a matrix
	 The off-subspace noise is stored in COVARS.
    """
    
    mix = {}
    mix['type'] = 'gmm'
    mix['nin'] = dim
    mix['ncentres'] = ncentres
    mix['covar_type'] = covar_type  #spherical
    mix['priors'] = np.ones((1,ncentres))/(ncentres)
    mix['centres'] = np.random.randn(ncentres, dim)
    mix['covars'] = np.ones((1, ncentres))
    mix['nwts'] = ncentres + ncentres*dim + ncentres
    return mix

def gmmactiv(mix, x):
    """Description
	This function computes the activations A (i.e. the  probability
	P(X|J) of the data conditioned on each component density)  for a
	Gaussian mixture model.  For the PPCA model, each activation is the
	conditional probability of X given that it is generated by the
	component subspace. The data structure MIX defines the mixture model,
	while the matrix X contains the data vectors.  Each row of X
	represents a single vector.
    """
    
    ndata = np.shape(x)[0]
    a = np.zeros(ndata, mix['ncentres'])
    n2 = np.zeros((ndata, mix['ncentres']), dtype=np.float64)

    for i in range(ndata):
        for j in range(mix['ncentres']):
            n2[i, j] = np.linalg.norm(x[i, :] - mix['centres'][j, :])

    # n2 = dist2(x, mix['centres']) # dist2
    # n2 = np.array(n2)
    wi2 = np.ones((ndata,1))*  (2* mix['covars'])
    nr, nc = np.shape(wi2)
    for i in range(0, nr):
        for j in range(0, nc):
            if wi2[i,j] == 0:
               wi2[i,j] = np.eps
    
    normal = (np.pi * wi2)**(mix['nin']/2)
    a = np.exp((-n2/wi2)/normal)
    return a

def gmmpost(mix,x):
    """	Description
	This function computes the posteriors POST (i.e. the probability of
	each component conditioned on the data P(J|X)) for a Gaussian mixture
	model.   The data structure MIX defines the mixture model, while the
	matrix X contains the data vectors.  Each row of X represents a
	single vector.
    """

    ndata = np.shape(x)[0]
    a = gmmactiv(mix, x)
    old_post = np.ones((ndata,1))*mix['priors']*a
    s = np.sum(old_post, axis = 1).reshape((-1, 1)) # s = sum(post,2)
    post = old_post/(np.matmul(s,np.ones((1,mix['ncentres']))) + 0.000001)
    post = np.array(post)
                     
                     
    return [post, a]    


def gmmem(x, mix, options = None):
    """Description
	[MIX, OPTIONS, ERRLOG] = GMMEM(MIX, X,) uses the Expectation
	Maximization algorithm of Dempster et al. to estimate the parameters
	of a Gaussian mixture model defined by a data structure MIX. The
	matrix X represents the data whose expectation is maximized, with
	each row corresponding to a vector. The optional parameters have
	the following interpretations.
    """


    v = np.array([], dtype = 'f')
    ndata = np.shape(x)[0]
    for n in range(0,100):
        [post, act] = gmmpost(x,mix)
        
        #adjust the new estimate for the parameters
        new_pr = np.sum(post, axis = 0)
        new_c = post*x
        
        #itetrate
        mix['priors'] = new_pr/ndata
        mix['centres'] = new_c/(new_pr * np.ones((1,mix['nin'])))
        if mix['covar_type'] == 'spherical':
            n2 = np.linalg.norm(x - mix['centres'])
            
            for i in range(0, mix['ncentres']):
                v_i = (post[:,i])*(n2[:,i])
                v.append(v_i)
            mix['covars'] = np.array([(var/new_pr)/mix['nin'] for var in v])
            
    return mix 


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x)) #axis = 0

def dmm(dim, ncentres, dist_type, nvalues, a = None, b = 1):
    """The function has the returns the same result as the gmm, 
    but for discrete type of data"""

    mix = {}
    mix['type'] = 'dmm'
    mix['input_dim'] = dim
    mix['ncentres'] = ncentres
    mix['dist_type'] = dist_type
    mix['priors'] = np.ones((1, ncentres)) / ncentres
    if dist_type == 'bernoulli':
        mix['nvalues'] = 1
        mix['nin'] = dim
        mix['means'] = np.random.rand(ncentres, dim)
        # if a < 1:
        #     print('a gives a singular prior')
        # else:
        #     mix['a'] = a
        # if b < 1:
        #     print('b gives a singular prior')
        # else:
        #     mix['b'] = b
            
    elif dist_type == 'multinomial':
        mix['nvalues'] = nvalues
        mix['nin'] = np.sum(nvalues)
        mix['means'] = np.zeros((ncentres, dim))
        k = 0
        # a = np.shape(mix['nvalues'][1])
        for i in range(0, len(mix['nvalues'])):
            mix['means'][:,k:k+mix['nvalues'][i]] = softmax(np.random.randn(ncentres,nvalues[i]))
            k = mix['nvalues'][i]
            # if a < 1:
            #     print('a gives a singular prior')
            # else :
            #     mix['a'] = a
    else:
        print('unknown distribution.')



def dmmactiv(mix,x):
    """active function for discrete type of data."""
    ndata = np.shape(x)[0]
    a = np.zeros((ndata, mix['ncentres']))
    e = np.ones((ndata,1))
    if mix['dist_type'] == 'bernoulli':
        for m in range(mix['ncentres']):
            a[:,m] =  np.prod(((e*(mix['means'][m,:]))**x)*
                 ((e*(1-mix['means'][m,:]))**(1-x)), 2)
    elif mix['dist_type'] == 'multinomial':
        for m in range(mix['ncentres']):
            a[:,m] = np.prod(((e*(1-mix['means'][m,:]))**(1-x)), 2)
    else:
        print('unknown distribution type.')

def dmmpost(mix,x):
    """return the posterior of discrete data"""
    a = dmmactiv(mix,x)
    ndata = np.shape(x)[0]
    post = (np.ones(ndata)[0]*mix['priors']*a)
    s = np.sum(post, axis = 0)
    post = post/(s*np.ones((1,mix['ncenres'])))
    return post



    