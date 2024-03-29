import pandas as pd
import numpy as np
import sys
from utilities import *

#Constructing rbf network
def rbf(nin, nhidden, nout, rbfunc, outfunc , prior = None, beta = None):
    """Create a net structure of RBF, we use dictionary.
       nin, number of inputs
       nhidden, number of hidden units
       nout, number of outputs
       nwts, total number of weights and biases
       actfn, activation function for hidden unit
       outfn, output error function
       prior, inverse variance value of a zero-mean Gaussian
       beta, inverse noise variance
    """
    rbf_net = {}
    rbf_net['nin'] = nin
    rbf_net['nhidden'] = nhidden
    rbf_net['nout'] = nout
    rbf_net['actfn'] = rbfunc
    rbf_net['outfunc'] = outfunc
    rbf_net['nwts' ] = nin*nhidden + (nhidden + 1)*nout + nhidden
    rbf_net['prior'] = prior
    rbf_net['beta'] = beta
    rbf_net['wi'] = np.ones(nhidden)
    rbf_net['alpha'] = prior
    w = np.random.randn(1, rbf_net['nwts'])
    net = rbfunpak(rbf_net, w)

    return net

def rbfunpak(net, w):
    """
    Description
	NET = RBFUNPAK(NET, W) takes an RBF network data structure NET and  a
	weight vector W, and returns a network data structure identical to
	the input network, except that the centres C, the widths WI, the
	second-layer weight matrix W2 and the second-layer bias vector B2
	have all been set to the corresponding elements of W.
    """
    nin = net['nin']
    nhidden = net['nhidden']
    nout = net['nout']
    mark1 = nin*nhidden
    net['c'] = np.reshape(w[0,0:mark1],(nhidden,nin))
    mark2 = mark1 + nhidden
    net['wi'] = np.reshape(w[0, mark1:mark2],(1,nhidden))
    mark3 = mark2 + nhidden*nout
    net['w2'] = np.reshape(w[0,mark2:mark3],(nhidden, nout))
    mark4 = mark3 + nout
    net['b2'] = np.reshape(w[0,mark3:mark4],(1,nout))
    return net



def rbfsetfw(net, scale):
    """Description
	NET = RBFSETFW(NET, SCALE) sets the widths of the basis functions of
	the RBF network NET. If Gaussian basis functions are used, then the
	variances are set to the largest squared distance between centres if
	SCALE is non-positive and SCALE times the mean distance of each
	centre to its nearest neighbour if SCALE is positive.  Non-Gaussian
	basis functions do not have a width.
    """
    real_max = sys.float_info.max
    cdist =  dist2(net['c'] , net['c'])
    # print(cdist)
    if scale > 0:
        cdist = cdist + real_max* np.eye(net['nhidden'])
        widths = scale*np.mean(np.min(cdist))
    else:
        widths = max(max(cdist))

    net['wi'] = widths * np.ones(np.shape(net['wi']))
    return net


def rbffwd(net, x):
    """Description
	A = RBFFWD(NET, X) takes a network data structure NET and a matrix X
	of input vectors and forward propagates the inputs through the
	network to generate a matrix A of output vectors. Each row of X
	corresponds to one input vector and each row of A contains the
	corresponding output vector. The activation function that is used is
	determined by NET.ACTFN.

	[A, Z, N2] = RBFFWD(NET, X) also generates a matrix Z of the hidden
	unit activations where each row corresponds to one pattern. These
	hidden unit activations represent the design matrix for the RBF.  The
	matrix N2 is the squared distances between each basis function centre
	and each pattern in which each row corresponds to a data point.
    """

    ndata = np.shape(x)
    n2 = np.zeros((ndata[0], net['c'].shape[0]), dtype=np.float64)
    for i in range(ndata[0]):
        for j in range(net['c'].shape[0]):
            n2[i, j] = np.linalg.norm(x[i, :] - net['c'][j, :])
    wi2 = np.ones((ndata[0],1))*(2*net['wi'])
    z = np.exp(-(n2/wi2))
    a = np.matmul(z,net['w2']) + np.ones((ndata[0],1))*net['b2']
    return a, n2


def rbfprior(rbfunc, nin, nhidden, nout):
    """Description
	[MASK, PRIOR] = RBFPRIOR(RBFUNC, NIN, NHIDDEN, NOUT, AW2, AB2)
	generates a vector MASK  that selects only the output layer weights.
	This is because most uses of RBF networks in a Bayesian context have
	fixed basis functions with the output layer as the only adjustable
	parameters.  In particular, the Neuroscale output error function is
	designed to work only with this mask.

	The return value PRIOR is a data structure,  with fields PRIOR.ALPHA
	and PRIOR.INDEX, which specifies a Gaussian prior distribution for
	the network weights in an RBF network. The parameters AW2 and AB2 are
	all scalars and represent the regularization coefficients for two
	groups of parameters in the network corresponding to  second-layer
	weights, and second-layer biases respectively. Then PRIOR.ALPHA
	represents a column vector of length 2 containing the parameters, and
	PRIOR.INDEX is a matrix specifying which weights belong in each
	group. Each column has one element for each weight in the matrix,
	using the standard ordering as defined in RBFPAK, and each element is
	1 or 0 according to whether the weight is a member of the
	corresponding group or not.
    """
    nwts_layer2 = nout + (nhidden *nout)
    if rbfunc == ['gaussian']:
        nwts_layer1 = nin*nhidden + nhidden
    else:
        print('Undefined activation function')
    nwts = nwts_layer1 + nwts_layer2
    mask = np.concatenate((np.zeros((nwts_layer1, 1)), np.ones((nwts_layer2, 1))), axis=0)
    # indx = np.zeros((nwts, 2))
    # mark2 = nwts_layer1 + (nhidden * nout)
    # indx[nwts_layer1:mark2, 0] = np.ones((nhidden * nout,))
    # indx[mark2:nwts, 1] = np.ones((nout,))
    # prior ={'index':indx}
    # return mask, prior
    return mask
