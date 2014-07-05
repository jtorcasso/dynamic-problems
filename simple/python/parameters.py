import numpy as np
from randomwalkincome import randomwalkincome

# Model Parameters
N = 20		# number of grid points of assets
I = 10		# number of grid points of permanent income
T = 45		# number of periods total
R = 15		# number of periods in retirement

# Income
inc_work = 15000*np.ones((1, T-R))
inc_ret = 5000*np.ones((1,R))
inc = np.hstack((inc_work, inc_ret))

# Labor Supply
work = np.hstack((np.ones((1, T-R)), np.zeros((1, R))))

# Random Income Component
sigma = 0.5
TransP, RI = randomwalkincome(I, sigma)
RandInc = np.exp(RI)
# Income variables

inc_unc = np.ones((I, T))
for i in xrange(I):
	print
	for t in xrange(T):
		inc_unc[i,t] = max(0, inc[0,t]*RandInc[i])

# Other Parameters
beta = 0.98
r = 0.03
gamma = 1.5
psi = 0.003

# Wealth Grid
WealthGrid = np.linspace(-200000, 2000000, N).reshape((N, 1))

#Find where grid is 0
a0 = abs(WealthGrid).argmin()