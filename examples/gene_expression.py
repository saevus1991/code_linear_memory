import sys
import numpy as np 
import matplotlib.pyplot as plt

pyssa_path = '/Users/christian/Documents/Code/pyssa'
sys.path.append(pyssa_path)
import pyssa.ssa as ssa
from pyssa.models.reaction_model import ReactionModel

# set model parameters
k_p = 5.0
T = 5.0
k_m = 1.0
n = 2.1
k_b = 1.0
gamma = 250.0
rates = np.array([gamma, k_m, T, k_p])
rates = np.concatenate([rates, rates, rates])
rates = np.ones(12)

# set propensity
S = np.array([
    [1, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, -1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, -1],
])

# propensity function 
m_ind = np.array([0, 2, 4])
p_ind = np.array([1, 3, 5])
p_ind_reac = np.array([5, 1, 3])
def prop(state):
    prop = np.ones(12)
    # mrna production
    prop[2*m_ind] = gamma/(1+k_b*state[p_ind_reac]**n)
    # mrna decay
    prop[2*m_ind+1] = k_m*state[m_ind]
    # protein prodcution
    prop[2*p_ind] = T*state[m_ind]
    # protein decay
    prop[2*p_ind+1] = k_p*state[p_ind]
    return(prop)


# set up model
model = ReactionModel(S, prop, rates)

# prepare simulation
tspan = np.array([0.0, 150.0])
initial = np.zeros(6)

# simulate 
simulator = ssa.Simulator(model, initial)
trajectory = simulator.simulate(initial, tspan, get_states=True)
t_grid = np.linspace(tspan[0], tspan[1], 500)
states_grid = ssa.discretize_trajectory(trajectory, t_grid)

# plot
plt.plot(t_grid, states_grid[:, 1], '-r')
plt.plot(t_grid, states_grid[:, 3], '-b')
plt.plot(t_grid, states_grid[:, 5], '-g')
plt.xlim([100, 150])
plt.show()