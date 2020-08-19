import sys
import numpy as np 
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from torchdiffeq import odeint_adjoint as odeint

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
from pymbvi.util import autograd_jacobian
from pymbvi.models.mjp.autograd_partition_specific_models import SimpleGeneExpression


class LinearODE(torch.nn.Module):

    def __init__(self, A, b):
        super(LinearODE, self).__init__()
        self.A = torch.nn.Parameter(A.clone())
        self.b = torch.nn.Parameter(b.clone())

    def forward(self, time, state):
        dydt = self.A @ state + self.b
        return(dydt)


# fix seed
np.random.seed(2007141048)

# prepare  model for simulation
pre, post, rates = sm.get_standard_model("simple_gene_expression")
pre = np.array(pre, dtype=np.float64)
post = np.array(post, dtype=np.float64)
rates = np.array(rates)
initial = np.array([0.0, 1.0, 0.0, 0.0])
tspan = np.array([0.0, 5000.0])
t_plot = np.linspace(tspan[0], tspan[1], 20)

# number of trajectories
num_samples = 10000

# compute ODE mean
rre_model = RREModel(pre, post, rates)
def odefun(time, state):
    return(rre_model.eval(state, time) )
sol = solve_ivp(odefun, tspan, initial, t_eval=t_plot)
states_rre = sol['y'].T

# set up moment model
moment_initial = np.zeros(9)
moment_initial[0:3] = initial[1:4]
mbvi_model = SimpleGeneExpression(moment_initial, np.log(rates), tspan)
moment_initial = torch.tensor(moment_initial)
def fun(state):
    tmp = mbvi_model.forward_torch(0.0, state, torch.zeros(rates.shape), mbvi_model.rates)
    return(tmp)
A_true = autograd_jacobian(fun, moment_initial)
b_true = fun(torch.zeros(moment_initial.shape))
model = LinearODE(A_true, b_true)

# compute moment solution
with torch.no_grad():
    states_mb = odeint(model, moment_initial, torch.from_numpy(t_plot))

# set up an observation model
sigma = np.array([0.15])
num_species = 4
obs_species = 3
obs_model = LognormObs(sigma, num_species, obs_species, num_species-1, obs_species-1)
delta_t = 300.0
t_obs = np.arange(tspan[0]+0.5*delta_t, tspan[1], delta_t)

# store for data
data = {}
data['rates'] = rates
data['initial'] = initial
data['moment_initial'] = moment_initial.numpy()
data['A_true'] = A_true.numpy()
data['b_true'] = b_true.numpy()
data['num_samples'] = num_samples
data['tspan'] = tspan
data['t_plot'] = t_plot
data['delta_t'] = delta_t
data['t_obs'] = t_obs
data['states_rre'] = states_rre
data['states_mb'] = states_mb

for i in range(num_samples):

    # get trajectory
    seed = np.random.randint(2**16)
    trajectory = gillespie.simulate(pre, post, rates, initial, tspan, seed)

    # get a subsampling for plotting
    states_plot = ssa.discretize_trajectory(trajectory, t_plot)

    # produce observations 
    observations = ssa.discretize_trajectory(trajectory, t_obs, obs_model=obs_model)

    # save
    data['states_plot_{}'.format(i)] = states_plot
    data['observations_{}'.format(i)] = observations

# save
save_path = '/Users/christian/Documents/Code/linear_memory/data/gene_expression/data.npz'
np.savez(save_path, **data)
