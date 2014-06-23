from parameters import *
from utility import utility

# Value and Policy Functions
V = np.zeros((N, T, I))
Policy = np.zeros((N, T, I))
EV = np.zeros((N, T, I))

# Terminal Period
cons = np.tile(WealthGrid, (1, I)) + np.tile(inc_unc[:,T-1].T, (N, 1))
V[:,T-1] = utility(cons, work[:,T-1])
Policy[:,T-1] = a0
# Other Periods

for t in xrange(T-2, -1, -1):
	for i in xrange(I):
		ev = np.zeros(N)
		for ii in xrange(I):
			ev += TransP[ii, i]*(V[:,t+1][:,ii])
		ev[np.isnan(ev)] = -np.inf
		EV[:,t+1][:,i] = ev
	vvv = np.transpose(np.tile(EV[:,t+1], (1,N,1)), (1,0,2))
	cons = np.transpose(np.tile(WealthGrid, (1,N,I)), (0,1,2)) + \
	       np.transpose(np.tile(inc_unc[:,t], (1,N,N)), (1,2,0)) + \
	       np.transpose(np.tile(WealthGrid/(1+r), (1,N,I)), (1,0,2))
	objective = utility(cons, work[t]) + beta*vvv
	V[:,t], Policy[:,t] = objective.max(axis=1, keepdims=True)

Policy[~np.isfinite(V)] = np.nan