import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

T = 45
R = 15
N = 1000          # Number of grid points for assets
I = 10          # Number of grid points for random income component
B = 2           # Possible assets liquidation

gamma = 1.5
psi   = 0.003
beta = 0.98
r = 0.03

def utility(C, L):
    '''utility function
    
    Parameters
    ----------
    C : array
        values of consumption
    L : array
        dichotomous labor supply
    
    Returns
    -------
    U : array
        level of utility
    '''
    
    U = (C**(1-gamma))/(1-gamma) - psi*L
    
    U[C <= 0.00000001] = -np.inf
    
    return U

def randomwalkincome(points, sigma):
    '''markov process of random component
    
    Parameters
    ----------
    points : int
        number of grid points for income process
    sigma : float
        standard deviation for the transition
        matrix
    
    Returns
    -------
    PROB : array
        transition matrix from shock i to j
    e : array
        random shocks to income
    '''
    
    e = np.linspace(-3, 3, points)
    
    avg = (e[-1] - e[-2])/2.
    
    PROB = np.zeros((points, points))
    
    for i in xrange(points):
        for j in xrange(points):
            diff = abs(e[i] - e[j])
            PROB[i,j] = norm.cdf(diff+avg, 0, sigma) - \
                norm.cdf(diff-avg, 0, sigma)
    
    PROB /= PROB.sum(axis=0)
    
    return PROB, e

# Income Grid
inc = np.hstack((15000*np.ones((1,T-R)), 5000*np.ones((1,R))))

sigma = 0.5
PROB, e = randomwalkincome(I, sigma)
inc = np.outer(np.exp(e), inc)        # All possible values of income, all time periods

# Labor Supply
L = np.hstack((np.ones((1, T-R)), np.zeros((1, R))))

# Wealth Grid (total assets)
WealthGrid= np.linspace(-200000, 2000000, N).reshape((N, 1))
a0 = abs(WealthGrid).argmin()
WealthGrid = WealthGrid[a0:]
N = len(WealthGrid)

# Unliquidated Assets
AssetValue = np.zeros(T)
AV0 = 0
AV1 = 1000
AV2 = -25
for t in xrange(T):
    AssetValue[t] = max(AV0 + AV1*t + AV2*(t**2), 0)

# Value Function Matrix
V = np.zeros((N, T, I, B))

# Expected Value Function Matrix
EV1 = np.zeros((N, T, I))
EV2 = np.zeros((N, T, I))

# Policy Function Matrix
P = np.zeros((N, T, I, B))

# Storing Selling Decision
Sell = np.zeros((N, T, I, B))


# Recursive Solution
# ------------------

C2 = np.tile(WealthGrid, (1, I)) + np.tile(inc[:,T-1].T, (N, 1))
V[:,T-1,:,1] = utility(C2, L[:,T-1])
P[:,T-1,:,1] = a0
C1 = np.tile(WealthGrid, (1, I)) + np.tile(inc[:,T-1].T, (N, 1))
V[:,T-1,:,0] = utility(C1, L[:,T-1])
P[:,T-1,:,0] = a0
Sell[:,T-1,:,0] = 1

for t in xrange(T-2, -1, -1):
    
    # Obtaining expected value of value function
    # in future for each possible income state i

    # ------------- IF SOLD ------------------#
    for i in xrange(I):
        ev = np.zeros(N)
        for j in xrange(I):
            ev += PROB[j, i]*V[:,t+1,j,1]
        ev[np.isnan(ev)] = -np.inf
        EV2[:,t+1,i] = ev
    
    V_next2 = np.tile(EV2[:,t+1], (N,1,1))

    # --------IF NOT SOLD --------------------#
    for i in xrange(I):
        ev = np.zeros(N)
        for j in xrange(I):
            ev += PROB[j, i]*V[:,t+1,j,0]
        ev[np.isnan(ev)] = -np.inf
        EV1[:,t+1,i] = ev
    
    V_next1 = np.tile(EV1[:,t+1], (N,1,1))

    # Budget Constraint and Maximization

    
    A_t = np.transpose(np.tile(WealthGrid[:,:,np.newaxis], (1,N,I)), (0,1,2))
    Y_t = np.transpose(np.tile(inc[:,t][:,np.newaxis,np.newaxis], (1,N,N)), (1,2,0))
    A_t_1 = np.transpose(np.tile(WealthGrid[:,:,np.newaxis]/(1+r), (1,N,I)), (1,0,2))
    C_t_2 = A_t + Y_t - A_t_1                   # If Sold
    C_t_1 = A_t + Y_t - A_t_1 + AssetValue[t]   # If Not Sold

    # ------------- IF SOLD ------------------#
        
    objective = utility(C_t_2, L[0,t]) + beta*V_next2
    V[:,t,:,1] = np.squeeze(objective.max(axis=1))
    P[:,t,:,1] = np.squeeze(objective.argmax(axis=1))

    # --------IF NOT SOLD --------------------#
    # -- Sell
    objective = utility(C_t_1, L[0,t]) + beta*V_next2
    v1 = np.squeeze(objective.max(axis=1))
    p1 = np.squeeze(objective.argmax(axis=1))

    # -- Do Not Sell
    objective = utility(C_t_2, L[0,t]) + beta*V_next1
    v2 = np.squeeze(objective.max(axis=1))
    p2 = np.squeeze(objective.argmax(axis=1))

    V[:,t,:,0] = np.maximum(v1, v2)
    Sell[:,t,:,0] = v1 > v2
    P[:,t,:,0] = (v1 > v2)*p1 + (v1<=v2)*p2
    
P[~np.isfinite(V)] = np.nan


# Simulation
# ----------

K = 5000

# Simulating Income Shocks
norm_draws = np.random.randn(T, K)
value = np.zeros((T, K))
value[0, :] = norm_draws[0, :]

# Adding Autocorrelation to income shocks
for t in xrange(1, T):
    value[t, :] = value[t-1, :] + norm_draws[t, :]

# Finding out where our simulated data comes close
# to the data we had in our grid
distI = np.zeros((T, K))
for t in xrange(T):
    for k in xrange(K):
        distI[t, k] = np.argmin(abs(e - value[t, k]))

distI = distI.astype('int')

# Simulating Assets (the state) by 
# selecting the value of assets
# associated with our level of simulated
# income. Thus, we end up with the path
# of asset choices (i.e. from t=1,...,T)
# for all K paths of income shocks
Liquidate = np.ones((T, K))
state = np.zeros((T, K))
state[0, :] = P[a0+1, t, distI[0, :], 0]
for t in xrange(1, T):
    for k in xrange(K):
        if Liquidate[t-1, k] == 2:
            state[t, k] = P[state[t-1, k], t, distI[t, k], 1]
            Liquidate[t, k] = 2
        else:
            state[t, k] = P[state[t-1, k], t, distI[t, k], 0]
            if Sell[state[t-1, k], t, distI[t, k], 0] == 1:
                Liquidate[t, k] = 2
            else:
                Liquidate[t, k] = 1

state = state.astype('int')
WealthProfile = WealthGrid[state]
Liquidate = Liquidate - 1

assets = WealthProfile.mean(axis=1)
rate = Liquidate.mean(axis=1)

# plt.plot(xrange(T), assets)
plt.plot(xrange(T), rate)
plt.show()
