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
	# Budget constraint
	vvv = np.tile(EV[:,t+1], (N,1,1))
	c1 = np.tile(WealthGrid, (N,1,I))
	c2 = np.tile(inc_unc[:,t], (N,N,1))
	c3 = np.tile(WealthGrid/(1+r), (N,1,I))
	cons = c1 + c2 + c3
	
	objective = utility(cons, work[0,t]) + beta*vvv
	V[:,t] = np.squeeze(objective.max(axis=1, keepdims=True))
	Policy[:,t] = np.squeeze(objective.argmax(axis=1))
	print 'Period {} Completed.'.format(t)

Policy[~np.isfinite(V)] = np.nan