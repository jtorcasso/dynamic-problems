from parameters import gamma, psi
from numpy import inf

def utility(x,part):

	gamma1 = 1 - gamma
	y = ((x**gamma1)/gamma1) - psi*part
	y[x <= 0.00000001] = -inf

	return y