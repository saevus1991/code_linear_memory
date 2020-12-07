import sys
import os
import numpy as np 
import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from import_utils import add_path

add_path('pyssa')
#pyssa_path = '/Users/christian/Documents/Code/pyssa'
#sys.path.append(pyssa_path)
import pyssa.util as ut

torch.set_default_dtype(torch.float64)

# set up  model 
class LinearODE(torch.nn.Module):

    def __init__(self, A, b):
        super(LinearODE, self).__init__()
        self.A = torch.nn.Parameter(A.clone())
        self.b = torch.nn.Parameter(b.clone())

    def forward(self, time, state):
        dydt = self.A @ state + self.b
        return(dydt)

# load training result
load_path = os.path.dirname(os.path.realpath(__file__)) + '/learn_linear_ode_minibatch.pt'
data = torch.load(load_path)
epoch = data['epoch']
model_state_dict = data['model_state_dict']
optimizer_state_dict = data['optimizer_state_dict']    
loss_history = data['loss_history']
states_mean = data['states_mean']
states_err = data['states_err']

# load data
load_path = os.path.dirname(os.path.realpath(__file__)) + '/data.npz'
data = np.load(load_path)
moment_initial = torch.from_numpy(data['moment_initial'])
rates = data['rates']
A_true = data['A_true']
b_true = data['b_true']
num_samples = data['num_samples']
tspan = data['tspan']
t_plot = data['t_plot']
delta_t = data['delta_t']
t_obs = data['t_obs']
states_rre = data['states_rre']
states_mb = data['states_mb']
num_steps, num_species = data['states_plot_0'].shape

# set up model
model = LinearODE(torch.zeros(A_true.shape), torch.zeros(b_true.shape))
model.load_state_dict(model_state_dict)

# compute final
with torch.no_grad():
    states_final = odeint(model, moment_initial, torch.from_numpy(t_plot))

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.plot(t_plot, states_mean[:, i], '-g')
    #plt.plot(t_plot, states_mb[:, i], '-r')
    plt.plot(t_plot, states_final[:, i], '-r')
    plt.fill_between(t_plot, states_mean[:, i] - states_err[:, i], states_mean[:, i] + states_err[:, i], color='g', alpha=0.2)
plt.show()


