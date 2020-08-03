import sys
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

pyssa_path = '/Users/christian/Documents/Code/pyssa'
sys.path.append(pyssa_path)
import pyssa.ssa as ssa
import pyssa.models.standard_models as sm
from pyssa.models.reaction_model import ReactionModel
import pyssa.ssa_compiled.gillespie as gillespie
from pyssa.models.cle_model import RREModel

pymbvi_path = '/Users/christian/Documents/Code/pymbvi'
sys.path.append(pymbvi_path)
from pymbvi.models.observation.kinetic_obs_model import LognormObs

# fix seed
#np.random.seed(2007211010)

file_path = '/Users/christian/Documents/Code/pyssa/pyssa/models/collection/stochastic_repressilator.xlsx'
df = pd.ExcelFile(file_path).parse()

# load model
pre, post, rates = sm.get_standard_model("stochastic_repressilator")

# prepare initial conditions
initial = np.zeros(15)
initial[1] = 20
initial[2] = 0.1
initial[3] = 0
initial[6] = 20
initial[11] = 20
tspan = np.array([0.0, 50000*60])
t_plot = np.linspace(tspan[0], tspan[1], 200)

# compute rre solution
rre_model = RREModel(pre, post, rates)
def odefun(time, state):
    return( rre_model.eval(state, time) )
sol = solve_ivp(odefun, tspan, initial, t_eval=t_plot, method='BDF')
states_rre = sol['y'].T

# # plot deterministic
# plt.subplot(2, 2, 1)
# for i in range(3):
#     ind = 1+5*i
#     plt.plot(t_plot/60, states_rre[:, ind])
# plt.subplot(2, 2, 2)
# for i in range(3):
#     ind = 2+5*i
#     plt.plot(t_plot/60, states_rre[:, ind])
# plt.subplot(2, 2, 3)
# for i in range(3):
#     ind = 3+5*i
#     plt.plot(t_plot/60, states_rre[:, ind])
# plt.subplot(2, 2, 4)
# for i in range(3):
#     ind = 4+5*i
#     plt.plot(t_plot/60, states_rre[:, ind])
# plt.show()

# number of simulations
num_samples = 0

# set up an observation model
sigma = np.array([0.15])
num_species = 4
obs_species = 3
obs_model = LognormObs(sigma, num_species, obs_species, num_species-1, obs_species-1)
delta_t = 300.0
t_obs = np.arange(tspan[0]+0.5*delta_t, tspan[1], delta_t)

# store for data
data = {}
data['num_samples'] = num_samples
data['tspan'] = tspan
data['t_plot'] = t_plot
data['delta_t'] = delta_t
data['t_obs'] = t_obs
data['states_rre'] = states_rre

for i in range(num_samples):

    # get trajectory
    seed = np.random.randint(2**16)
    trajectory = gillespie.simulate(pre, post, rates, initial, tspan, seed, t_eval=t_plot)

    # produce observations 
    states_ssa = trajectory['states']
    #observations = ssa.discretize_trajectory(trajectory, t_obs, obs_model=obs_model)

    # save
    #data['states_ssa_{}'.format(i)] = states_ssa
    #data['observations_{}'.format(i)] = observations


# # save
# save_path = '/Users/christian/Documents/Code/linear_memory/data/gene_expression/data.npz'
# np.savez(save_path, **data)

# plot deterministic
colors = ['r', 'b', 'g']
plt.subplot(2, 2, 1)
for i in range(3):
    ind = 1+5*i
    #plt.plot(t_plot/60, states_ssa[:, ind], ':', color=colors[i])
    plt.plot(t_plot/60, states_rre[:, ind], '-', color=colors[i])
plt.subplot(2, 2, 2)
for i in range(3):
    ind = 2+5*i
    #plt.plot(t_plot/60, states_ssa[:, ind], ':', color=colors[i])
    plt.plot(t_plot/60, states_rre[:, ind], '-', color=colors[i])
plt.subplot(2, 2, 3)
for i in range(3):
    ind = 3+5*i
    #plt.plot(t_plot/60, states_ssa[:, ind], ':', color=colors[i])
    plt.plot(t_plot/60, states_rre[:, ind], '-', color=colors[i])
plt.subplot(2, 2, 4)
for i in range(3):
    ind = 4+5*i
    #plt.plot(t_plot/60, states_ssa[:, ind], ':', color=colors[i])
    plt.plot(t_plot/60, states_rre[:, ind], '-', color=colors[i])
plt.show()