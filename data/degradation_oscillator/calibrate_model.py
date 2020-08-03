import sys
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
np.random.seed(2007141048)

# prepare  model for simulation
pre, post, rates = sm.get_standard_model("degradation_oscillator")
pre = np.array(pre, dtype=np.float64)
post = np.array(post, dtype=np.float64)
rates = np.array(rates)
initial = np.array([0.0, 0.0, 0.0, 0.0])
tspan = np.array([0.0, 6*60*60])

# number of trajectories
num_samples = 5000

# prepare initial conditions
t_plot = np.linspace(tspan[0], tspan[1], 200)

# compute ODE mean
rre_model = RREModel(pre, post, rates)
def odefun(time, state):
    return( rre_model.eval(state, time) )
sol = solve_ivp(odefun, tspan, initial, t_eval=t_plot)
states_rre = sol['y'].T

# plot 
plt.plot(t_plot/60, 100*states_rre[:, 0], '-k')
#plt.plot(t_plot, states_mean[:, 2], ':r')
plt.plot(t_plot/60, states_rre[:, 1], '-r')
#plt.plot(t_plot, states_mean[:, 3], ':b')
plt.plot(t_plot/60, states_rre[:, 2], '-b')
plt.plot(t_plot/60, states_rre[:, 3], '-g')
plt.show()

# # set up an observation model
# sigma = np.array([0.15])
# num_species = 4
# obs_species = 3
# obs_model = LognormObs(sigma, num_species, obs_species, num_species-1, obs_species-1)
# delta_t = 300.0
# t_obs = np.arange(tspan[0]+0.5*delta_t, tspan[1], delta_t)

# # store for data
# data = {}
# data['num_samples'] = num_samples
# data['tspan'] = tspan
# data['t_plot'] = t_plot
# data['delta_t'] = delta_t
# data['t_obs'] = t_obs
# data['states_rre'] = states_rre

# for i in range(num_samples):

#     # get trajectory
#     seed = np.random.randint(2**16)
#     trajectory = gillespie.simulate(pre, post, rates, initial, tspan, seed)

#     # get a subsampling for plotting
#     states_plot = ssa.discretize_trajectory(trajectory, t_plot)

#     # produce observations 
#     observations = ssa.discretize_trajectory(trajectory, t_obs, obs_model=obs_model)

#     # save
#     data['states_plot_{}'.format(i)] = states_plot
#     data['observations_{}'.format(i)] = observations

# # save
# save_path = '/Users/christian/Documents/Code/linear_memory/data/gene_expression/data.npz'
# np.savez(save_path, **data)