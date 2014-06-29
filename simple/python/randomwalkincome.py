import numpy as np
from scipy.stats import norm 

def randomwalkincome(Z, sigma):
	'''value and matrix of Markov matrix

	Parameters
	----------
	Z : int
		number of possible values process can take
	sigma : float
		standard deviation of the permanent income shock

	Returns
	-------
	trans : array
		individual Markov matrix
	xxx : array
		vector
	'''

	xxx = np.linspace(-3, 3, Z)

	pe = (xxx[-1] - xxx[-2])/2

	trans = np.zeros((Z,Z))

	for i in xrange(Z):
		for j in xrange(Z):
			diff = abs(xxx[i] - xxx[j])
			trans[i,j] = norm.cdf(diff+pe, 0, sigma) - \
						 norm.cdf(diff-pe, 0, sigma)

	# trans /= trans.sum(axis=0)
	for i in xrange(Z):
		trans[:,i] /= trans[:,i].sum()

	return trans, xxx