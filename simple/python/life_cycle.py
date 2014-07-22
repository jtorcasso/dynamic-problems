
# The Economic World
# ==================
# 
# In this world, agents live for $T$ periods, the last $R$ periods of which are in retirement. 

# In[1]:

T = 45
R = 15


# Utility
# -------
# 
# \begin{equation}
#     U(C_t,L_t) = \frac{C_t^{1-\gamma}}{1-\gamma} - \psi L_t 
# \end{equation}
# 
# , where $0 \leq \gamma \leq 1$, $C_t$ is consumption and $L_t \in \{0,1\}$ is labor supply. Notice that we make consumption less than 0.00000001 carry extremely negative utility.

# In[2]:

import numpy as np

gamma = 1.5
psi   = 0.003

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


# Budget Constraint
# -----------------
# 
# \begin{equation}
#     C_t + \frac{A_{t+1}}{1+r} = Y_t + A_t
# \end{equation}
# 
# , where future and current assets are $A_{t+1}$ and $A_t$, and income is $Y_t$, with $C_t \geq 0$, $\forall t$. Upon death, the agent cannot leave any debt or bequests, i.e. $A_{T+1} = 0$. The earnings process is piecewise and taken to be exogenous.
# 
# Earnings Process
# ----------------
# 
# \begin{equation}
#  Y_t =
#   \begin{cases}
#    15000 e^{\epsilon_t} & \text{if } t < T-R \\
#    5000 e^{\epsilon_t}  & \text{if } t \geq T-R
#   \end{cases}
# \end{equation}
# 
# , where $\epsilon_t$ is drawn from a markov process with transition probabilities:
# 
# \begin{equation}
#     PROB(i,j) = \Phi_{\sigma}(\text{abs}(\epsilon_i - \epsilon_j) + \Delta) - \Phi_{\sigma}(\text{abs}(\epsilon_i - \epsilon_j) + \Delta)
# \end{equation}
# 
# , where $\Phi_{\sigma}$ is the normal cdf with standard deviation $\sigma$. 

# In[3]:

from scipy.stats import norm

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
            PROB[i,j] = norm.cdf(diff+avg, 0, sigma) -                         norm.cdf(diff-avg, 0, sigma)
    
    PROB /= PROB.sum(axis=0)
    
    return PROB, e


# The Agent's Problem
# ===================
# 
# The agent maximizes her expected lifetime discounted utility subject to a budget constraint, choosing the optimal asset levels. 
# 
# \begin{align}
#     V(t) & = \text{max}_{A_1,...,A_T} \sum_{t=1}^T \beta^{t-1}\mathcal{E}[U(C_t, L_t)] \\
#          & = \text{max}_{A_1,...,A_T} \mathcal{E}[U(C_1, L_1)] + \sum_{t=2}^T \beta^{t-2}\mathcal{E}[U(C_t, L_t)] \\
#          & = \text{max}_{A_1,...,A_T} \mathcal{E}[U(C_1, L_1)] + \beta \sum_{t=2}^T \beta^{t-1}\mathcal{E}[U(C_t, L_t)] \\
#          & = \text{max}_{A_1,...,A_T} \mathcal{E}[U(C_1, L_1)] + \beta V(t+1) \\
#          & = \text{max}_{A_1,...,A_T} U(C_1, L_1) + \beta V(t+1) \\
# \end{align}
# 
# s.t.
# 
# \begin{align}
#     C_t + \frac{A_{t+1}}{1+r} = Y_t + A_t \\
#           C_t & \geq 0 \\
#           A_{T+1} & = 0
# \end{align}
# 
# , where the expectation is taken with respect to the random component of the income process. 
# 
# Initialization
# --------------
# We define the support of income and assets as a discrete space. The permanent component of income is fixed while working and in retirement at 15000 and 5000, respectively. The random component of income is a multiplicative shock that follows a markov process with a discrete state space.

# In[4]:

N = 1000          # Number of grid points for assets
I = 10          # Number of grid points for random income component

# Income Grid
inc = np.hstack((15000*np.ones((1,T-R)), 5000*np.ones((1,R))))

sigma = 0.5
PROB, e = randomwalkincome(I, sigma)
inc = np.outer(np.exp(e), inc)        # All possible values of income, all time periods

# Labor Supply
L = np.hstack((np.ones((1, T-R)), np.zeros((1, R))))

# Wealth Grid (assets)
WealthGrid= np.linspace(-200000, 2000000, N).reshape((N, 1))
a0 = abs(WealthGrid).argmin()


# Next, we setup the value function and expected value function matrices. Then we setup the policy function matrix. Notice the large number of dimensions to account for all possible combinations along the support of the state variable (assets) and the income process. We will use these matrices to store our data as we move recursively through the dynamic programming problem.

# In[5]:

# Value Function Matrix
V = np.zeros((N, T, I))

# Expected Value Function Matrix
EV = np.zeros((N, T, I))

# Policy Function Matrix
P = np.zeros((N, T, I))

beta = 0.98
r = 0.03


# Recursive Solution
# ------------------
# To solve this problem we work backwards, starting with the terminal period. For this period we need not compute any expected values, as the agent knows she is dying and consumes everything. Then we proceed to the T-1 period, beginning our recursive solution, which proceeds as follows:
# 
#     1. Compute the expected value (for each income state i of I) in period T of the value function by taking the weighted average of V(T), weighting by the transition probabilities.
#     2. Then we obtain the consumption level at time T-1 using the budget constraint and our known support of income and assets.
#     3. We repeat these matrices so we have a matrix for each value along the support of the state. We do this because after
#     maximizing we need the value matrix at time T-1 to have shape (N,I). That is, we pick the best next state given all possible levels of our current state.
#     4. Then we maximize the current state's value function along the support of the state and find the associated index for the policy function [JT: Shouldn't we put the value of assets in the policy function, not the index?]

# In[6]:

C = np.tile(WealthGrid, (1, I)) + np.tile(inc[:,T-1].T, (N, 1))
V[:,T-1] = utility(C, L[:,T-1])
P[:,T-1] = a0


# Out[6]:

#     -c:22: RuntimeWarning: invalid value encountered in power
# 

# The error above occured because we have some infeasible consumption levels in our grid space, but we ensure these are set to null in our utility function. Now we can solve for the other periods going backwards from the terminal period.

# In[7]:

for t in xrange(T-2, -1, -1):
    
    # Obtaining expected value of value function
    # in future for each possible income state i
    for i in xrange(I):
        ev = np.zeros(N)
        for j in xrange(I):
            ev += PROB[j, i]*V[:,t+1,j]
        ev[np.isnan(ev)] = -np.inf
        EV[:,t+1,i] = ev
    
    V_next = np.tile(EV[:,t+1], (N,1,1))
    
    # Budget Constraint
    A_t = np.transpose(np.tile(WealthGrid[:,:,np.newaxis], (1,N,I)), (0,1,2))
    Y_t = np.transpose(np.tile(inc[:,t][:,np.newaxis,np.newaxis], (1,N,N)), (1,2,0))
    A_t_1 = np.transpose(np.tile(WealthGrid[:,:,np.newaxis]/(1+r), (1,N,I)), (1,0,2))
    C_t = A_t + Y_t - A_t_1
    
    objective = utility(C_t, L[0,t]) + beta*V_next
    V[:,t] = np.squeeze(objective.max(axis=1))
    P[:,t] = np.squeeze(objective.argmax(axis=1))
    
P[~np.isfinite(V)] = np.nan


# Out[7]:

#     -c:8: RuntimeWarning: invalid value encountered in multiply
# 

# Simulation
# ----------
# Now we now what an agent will do at each time period given any level of assets and any realization of the income shock. Thus, by simulating income shocks and starting with our initial stock of assets, we can trace the agent's savings decisions across the life-cycle. We do so for 5000 agents, or specifically, 5000 realizations of life-cycle earnings shocks. Then we plot the asset profile for the average agent. 

# In[13]:

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
state = np.zeros((T, K))
state[0, :] = P[a0+1, t, distI[0, :]]
for t in xrange(1, T):
    for k in xrange(K):
        state[t, k] = P[state[t-1, k], t, distI[t, k]]

state = state.astype('int')
WealthProfile = WealthGrid[state]

assets = WealthProfile.mean(axis = 1)

get_ipython().magic(u'matplotlib inline')

import matplotlib.pyplot as plt
plt.plot(xrange(T), assets)




# Out[13]:

#     (1000, 1)
# 

#     [<matplotlib.lines.Line2D at 0x7f3b89d8bdd0>]

# image file:

# In[ ]:



