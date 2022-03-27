import numpy as np
import rbfnet as rn
import mixmodel as mm
from utilities import *

def ggtm(dim_latent, nlatent, dim_data_array, map, mix_array):

    """Description

	NET = GGTM(DIM_LATENT, NLATENT, DIM_DATA_ARRAY, MAP, MIX_ARRAY), takes the
	dimension of the latent space DIM_LATENT, the number of data points
	sampled in the latent space NLATENT, a vector containing the dimensions
   of each data space type DIM_DATA_ARRAY, a MAP and MIX_ARRAY structures
   specifying the parameters of the mapping function and mixture models.
   Each element of the MIX_ARRAY array corresponds to the mixture
   model associated to each observation space. An observation space
   represents a certain type of data (continuous, binary or categorical).

	The fields in NET are
	   type = 'ggtm'
	   dim_latent = dimension of latent space
       obs_array =  array of observation spaces
       X = sample of latent points

   The MAP structure is defined as follows
	  map.type = 'rbf' or 'gp'
	  map.func = RBF activation function or GP covariance function
	  map.outfunc = RBF output function
	  map.prior = scalar representing the inverse variance of the RBF prior distribution
	  map.ncentres = number of RBF centres

   Each element of the MIX_ARRAY has the following structure
	  mix.type = 'gmm' or 'dmm'
	  mix.covar_type = 'spherical', 'diag' or 'full'
	  mix.a and mix.b = optional scalars to compute the parameter of beta prior (cf dmm)
	  mix.cat_nvals = row vector containing the number of values for each categorical variable


    """
    net = {}
    net['type'] = 'ggtm'
    ndata_space = np.shape(dim_data_array)[0]

    if ndata_space != np.shape(mix_array)[0]:
        print("The length of MIX_ARRAY must match the number of data spaces.")

    net['dim_latent'] = dim_latent
    obs_array = []
    for i in range(0,ndata_space):
        obs = {}
        obs['nin'] = dim_data_array[i]

        if map['type'] == ['rbf']:
            prior = map['prior']

        obs['mapping'] = rn.rbf(dim_latent, map['ncentres'], dim_data_array[i],
                           map['func'], 'linear', prior)
        obs['mapping']['mask'] = rn.rbfprior(map['func'], dim_latent, map['ncentres'],
                               dim_data_array[i])
        if mix_array[i]['type'] == 'gmm':
            obs['type'] = 'continuous'
            obs['mix'] = mm.gmm(obs['nin'], nlatent, mix_array[i]['covar_type'])
        elif mix_array[i]['type'] == 'dmm':
            obs['type'] = 'discrete'
            obs['dist_type'] = mix_array[i]['dist_type']
            obs['mix'] = mm.gmm(obs['nin'], nlatent, mix_array[i]['cat_nvals'])
        else:
            print(' unknown mixture model')


        obs['covar_type'] = mix_array[i]['covar_type']
        # print(obs['nin'])
        obs_array.append(obs)

    net['X'] = []
    net['obs_array'] = obs_array
    return net

def ggtminitsubmodel(net,obs, dim_latent, X, data, samp_type, rbf_samp_size,varargin = None):


    """Description
	NET = GGTMINITSUBMODEL(NET, OBS, DIM_LATENT, X, OPTIONS, DATA, SAMP_TYPE,RBF_SAMP_SIZE)
   takes a GTM NET and initialises the parameters of each observation
   space. An observation space represents a certain type of data
   (continuous, binary or categorical). DIM_LATENT is the dimension of the
   latent space, X is sample of latent points and RBF_SAMP_SIZE is the
   size of the RBF specification for a RBF mapping function.

	If the SAMPTYPE is 'REGULAR', then regular grids of latent data
	points and RBF centres a re created.  The dimension of the latent data
	space must be 1 or 2.  For one-dimensional latent space, the
	LSAMPSIZE parameter gives the number of latent points and the
	RBFSAMPSIZE parameter gives the number of RBF centres.  For a two-
	dimensional latent space, these parameters must be vectors of length
	2 with the number of points in each of the x and y directions to
	create a rectangular grid.  The widths of the RBF basis functions are
	set by a call to RBFSETFW passing OPTIONS(7) as the scaling
	parameter.

	If the SAMPTYPE is 'UNIFORM' or 'GAUSSIAN' then the latent data is
	found by sampling from a uniform or Gaussian distribution
	correspondingly.  The RBF basis function parameters are set by a call
	to RBFSETBF with the DATA parameter as dataset and the OPTIONS
	vector.

	Finally, the output layer weights of the RBF are initialised by
	mapping the mean of the latent variable to the mean of the target
	variable, and the L-dimensional latent variale variance to the
	variance of the targets along the first L principal components.
    """


    data_mat = data['mat']
    nrow, ncol = np.shape(data_mat)

    nhidden = obs['mapping']['nhidden']
    if samp_type == 'regular':
        # ignore dim_latent = 1
        if dim_latent == 2:
            obs['mapping']['c'] = gtm_rctg(rbf_samp_size)
            obs['mapping'] = rn.rbfsetfw(obs['mapping'], 1)
            # print(obs['mapping']['w2'])

    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(data_mat)
    PCcoeff = pca.explained_variance_
    PCvec = pca.components_.T
    A = np.dot(PCvec[:,0:dim_latent],np.diag(np.sqrt(PCcoeff[0:dim_latent])))
    Phi = rn.rbffwd(obs['mapping'],X)[1]
    x_std_array = 1/np.std(X, axis=0, ddof=1)
    x_mean_diag = np.diag(np.mean(X,axis = 0))
    temp1 = X - np.dot(np.ones(np.shape(X)),x_mean_diag)
    normX = np.dot(temp1,np.diag(x_std_array))
    obs['mapping']['w2'] = np.linalg.lstsq(Phi,np.dot(normX,A.T),rcond=None)[0]

    obs['mapping']['b2'] = np.mean(data_mat, axis = 0)

    if obs['type'] == 'continuous':
        if varargin != None:
            from sklearn.model_selection import train_test_split
            samp_train_data, samp_test_data = train_test_split(data_mat,
                                                       test_size = 0.25, random_state=42)
            obs_pca = PCA(n_components = dim_latent)
            obs_pca.fit(samp_train_data)
            samp_pcvec = np.transpose(obs_pca.components_)
            projection = np.dot(samp_train_data ,PCvec)
            obs = gtm_2d_init(net, obs, projection, samp_train_data)
        else:
            obs['mix']['centres'] = rn.rbffwd(obs['mapping'], net['X'])

        import sys
        realmax = sys.float_info.max
        d = dist2(obs['mix']['centres'][0],obs['mix']['centres'][0]) + np.diag(np.ones(obs['mix']['ncentres'])*realmax)
        sigma = np.mean(np.min(d,axis = 0))/2
        # if dim_latent < ncol :
        #     sigma = min(sigma,PCcoeff[dim_latent+1])
        obs['covars'] = sigma*np.ones((1,obs['mix']['ncentres']))

    elif obs['type'] == 'discrete':
        if obs['dist_type'] == 'multinomial':
            for i in range(0, len(data['cat_nvals'])):
                W_tmp = ApplyPCAInitW(obs,data_mat[:,data['start_inds'][i]:data['end_inds'][i]],dim_latent,X)
                obs['mapping']['w2'][:,data['start_inds'][i]:data['end_inds'][i]] = W_tmp
    else:
        print('unkown data type.')

    return net,obs



def ApplyPCAInitW(obs,tempdata,dim_latent,X):

    """Apply PCA for initial Weight."""
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(tempdata)
    PCcoeff = pca.explained_variance_
    PCvec = np.transpose(pca.components_)
    A = np.dot(PCvec[:,0:dim_latent],np.diag(np.sqrt(PCcoeff[0:dim_latent])))
    Phi = rn.rbffwd(obs['mapping'],X)[1]
    normX = np.dot(np.dot(X - np.ones(np.shape(X)),np.diag(np.mean(X,axis=0))),np.diag(1/np.std(X, axis=0, ddof=1)))
    w2 = np.linalg.lstsq(Phi,np.dot(normX,A.T),rcond=None)[0]
    return w2


def ggtminit(net, data_array, samp_type , latent_shape, rbf_grid):

    """Description
	NET = GGTMINIT(NET, OPTIONS, DATA_ARRAY, SAMPTYPE) takes a Generalised
    GTM NET and generates a sample of latent data points. The array
    DATA_ARRAY contains the original data, which may have missing values,
    for each observation space. An observation space represents a certain
    type of data (continuous, binary or categorical).

	If the SAMPTYPE is 'REGULAR', then regular grids of latent data
	points are created.  The dimension of the latent data
	space must be 1 or 2.  For one-dimensional latent space, the
	LSAMPSIZE parameter gives the number of latent points.  For a two-
	dimensional latent space, these parameters must be vectors of length
	2 with the number of points in each of the x and y directions to
	create a rectangular grid.
    """

    ndata = np.shape(data_array)[0]
    nlatent = net['obs_array'][1]['mix']['ncentres']
    if samp_type == 'regular':
        l_samp_size = latent_shape
        rbf_samp_size = rbf_grid
        if net['dim_latent'] == 1:
            net['X'] = np.arange(-1,1,l_samp_size - 1)
        elif net['dim_latent'] == 2:
            net['X'] = gtm_rctg(l_samp_size)
            # for i in range (len(net['obs_array'])):
            #     net['obs_array'][i]['mapping']['c'] = gtm_rctg(rbf_samp_size)
            #     print(net['obs_array'][i]['mapping']['c'])
        else:
            print('For regular sample, input dimension must be 1 or 2.')
    else:
        print('Invalid sample type')

    nobs_space = len(net['obs_array'])
    if nobs_space != ndata:
        print("The number of observation spaces must match the number of data")

    for i in range (nobs_space):
        net,net['obs_array'][i] = ggtminitsubmodel(net, net['obs_array'][i],net['dim_latent'],net['X'],data_array[i],samp_type,rbf_samp_size)

    return net


from scipy import sparse
def ggtmem(net, t_array):
    """Description
	[NET, OPTIONS, ERRLOG, T_ARRAY] = GGTMEM(NET, T_ARRAY, OPTIONS, FSOPTIONS)
    uses the Expectation Maximization algorithm to estimate the parameters of
    a GGTM defined by a data structure NET. The array T_ARRAY contains the
    original data, which may have missing values for each observation space,
    whose expectation is maximized.
	It is assumed that the latent data NET.X has been set following a
	call to GGTMINIT, for example. The optional parameters have the
	following interpretations.
    """

    ntotaldata = np.shape(t_array)[0]
    nobs_space = ntotaldata
    niters = 100
    d_alpha = 0.001
    ND = np.zeros((1,nobs_space))
    K = np.zeros((1,nobs_space))
    var_array = np.array(([{'Phi':[],'PhiT':[],'A':[],'Alpha':[]},
                      {'Phi':[],'PhiT':[],'A':[],'Alpha':[]},
                      {'Phi':[],'PhiT':[],'A':[],'Alpha':[]}]))
    T_array = t_array
    for i in range(0,nobs_space):
        obs = net['obs_array'][i]
        t = t_array[i]['mat']
        ndata, tdim = np.shape(t)
        ND[0][i] = ndata*tdim
        if obs['type'] == 'continuous':
            var_array[i]['Phi'] = rn.rbffwd(obs['mapping'], net['X'])[2]
            one = np.ones((np.shape(net['X'])[0],1))
            PhiT = np.hstack((var_array[i]['Phi'],one))
            var_array[i]['PhiT'] = np.transpose(PhiT)
            K[0][i], Mplus1 = np.shape(var_array[i]['Phi'])
            var_array[i]['A'] = np.zeros((Mplus1, Mplus1))
            if obs['mapping']['alpha'] > 0 :
                eyeMat = sparse.eye(Mplus1).toarray()
                var_array[i]['Alpha'] = obs['mapping']['alpha']*eyeMat
                var_array[i]['Alpha'][Mplus1-1][Mplus1-1] = 0

        elif obs['type'] == 'discrete':
            var_array[i]['Phi'] = rn.rbffwd(obs['mapping'], net['X'])[2]
            one = np.ones((np.shape(net['X'])[0], 1))
            PhiT = np.hstack((var_array[i]['Phi'],one))
            var_array[i]['PhiT'] = np.transpose(PhiT)
            K[0][i], Mplus1 = np.shape(var_array[i]['Phi'])
            var_array[i]['K'] = K[0][i]
            obs['w'] = [obs['mapping']['w2'],obs['mapping']['b2']]
        else :
            print('Unknown noise model.')

    for j in range(0,niters):
        ninner = 100
        R = ggtmpost(net, t_array)[0]
        a_array = ggtmpost(net,t_array)[3]
        e = 0
        for k in range(0,nobs_space):
            obs = net['obs_array'][k]
            prob = a_array[:,:,k]*np.transpose((obs['mix']['priors']))
            e = e - np.sum(np.log(max(prob,np.spacing(1))))
            print(e)
        for m in range(0,nobs_space):
            obs = net['obs_array'][m]
            mstep_func = ['ggtm_mstep', obs['type']]
            [net, obs, T_array[i]] = eval(mstep_func, net,obs,t_array[i],
                                           R,var_array[i],
                                           K[i],ND[i],
                                           ninner ,
                                           d_alpha,T_array[i])



def ggtmlmean(net, data_array):
    """Description
    MEANS = GGTMLMEAN(NET, DATA_ARRAY) takes a Generalised GTM structure
    NET, and computes the means of the responsibility distributions for each data point in each data in DATA_ARRAY.

    """
    ntotaldata = np.shape(data_array)[1]
    nobs_space = np.shape(net['obs_array'])[1]
    R = ggtmpost(net,data_array)[0]
    a_array = ggtmpost(net,data_array)[3]
    means = R*net['X']

    lle = 0
    for i in range(nobs_space):
        obs = net['obs_array'][i]
        prob = a_array[:,:,i]*obs['mix']['priors']
        lle = lle - np.sum(np.log(max(prob, np.eps)))
    return lle, means


def ggtmpost(net, data_array):

    """
    """
    ntotaldata = np.shape(data_array)[0]
    nobs_space = np.shape(net['obs_array'])[0]
    a = np.shape(data_array[0]['mat'])[0]
    b = net['obs_array'][0]['mix']['ncentres']
    post_array = np.zeros(shape = (a, b, nobs_space))
    a_array = np.zeros(shape = np.shape(post_array))
    for i in range(0,nobs_space):
        obs = net['obs_array'][i]
        data = data_array[i]
        if obs['type'] == 'continuous':
            obs['mix']['centres'] = rn.rbffwd(obs['mapping'], net['X'])
            post_array[:,:,i], a_array[:,:,i] = mm.gmmpost(obs['mix'], data['mat'])
        elif obs['type'] == 'dicrete':
            obs['mix']['centres'], tmp_Phi = rn.rbffwd(obs['mapping'],net['X'])
            Phi = [tmp_Phi, np.ones([np.shape(net['X'])[0],1])]
            W = [obs['mapping']['w2'], obs['mapping']['b2']]
            if obs['dist_type'] == 'bernoulli':
                obs.mix.means = rn.inverselink(obs['dist_type'],Phi*W)
                post_array[:,:,i], a_array[:,:,i] = mm.dmmpost(obs['mix'],data['mat'])

            elif obs['dist_type'] == 'multinomial':
                for j in range(0,np.shape(data['cat_nvals'][1])):
                    PhiW = Phi*W[:,data.start_inds[j]:data.end_inds[j]]
                    obs['mix']['means'][:,data.start_inds[j]:data.end_inds[j]] = inverselink(obs['dist_type'],PhiW)
                post_array[:,:,i], a_array[:,:,i] = mm.dmmpost(obs['mix'],data['mat'])
            else:
                print('unknow discrete distribution type')

    return post, a,post_array, a_array, net



def ggtmjointpost(net, a_array):
    joint_post = np.zeros([np.shape(a_array)[0], np.shape(a_array)[1]])
    joint_lik = np.zeros((np.shape(joint_post)))
    joint_a = np.zeros((np.shape(joint_post)))

    nobs_space = np.shape(net['obs_array'])[1]

    for i in range(0, nobs_space):
        obs = net['obs_array'][i]
        if i == 1:
            joint_a = a_array[:,:,i]
            if obs['type'] == 'continuous':
                joint_lik = np.ones(np.shape([a_array[:,:,1]])[0],1)*obs['mix']['priors']*a_array[:,:,i]
            elif obs['type'] == 'discrete':
                joint_lik = a_array[:,:,i]
            else:
                print('unknown noise model')

        else :
            if obs['type'] == 'continuous':
                joint_lik = np.ones(np.shape([a_array[:,:,1]])[0],1)*obs['mix']['priors']*a_array[:,:,i]
                joint_a = joint_a*a_array[:,:,i]
            elif obs['type'] == 'discrete':
                joint_lik = a_array[:,:,i]
                joint_a = joint_a*a_array[:,:,i]
            else:
                print('unknown noise model')
    s = np.sum(joint_lik, axis = 0)
    joint_post = joint_lik / (s*np.ones((1, obs['mix']['ncentres'])))


import numpy.matlib
def inverselink(dist_type, x):
    if dist_type == 'bernoulli':
        y = 1/(1+np.exp(-x))

    elif dist_type == 'multinomial':
        x = np.tranpose(x)
        n = np.shape(x)[0]
        x = x - np.matlib.repmat(max(x),n,1)
        x = np.exp(x)
        x_sort = x.sort()
        y = x/(np.matlib.repmat(np.sum(x_sort),n,1))
        y = np.transpose(y)

    else:
        print('unkown distribution type')
