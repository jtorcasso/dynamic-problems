from parameters import gamma, psi
from numpy import inf

def utility(x,part):
	'''utility function

	U(C,L) = (1/(1-gamma))*C^(1-gamma) - psi*L

	Parameters
	----------
	x : array
		consumption
	part : array
		labor supply

	Returns
	-------
	y : array
		earnings
	'''

	gamma1 = 1 - gamma
	y = ((x**gamma1)/gamma1) - psi*part
	y[x <= 0.00000001] = -inf

	return y