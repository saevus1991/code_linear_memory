import numpy as np
import torch
import os
import matplotlib.pyplot as plt
#from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint
from linear_memory.linear_memory import LinearMemory
import linear_memory.utils as ut
from import_utils import add_path

add_path('pyssa')
import pyssa.ssa as ssa
import pyssa.models.standard_models as sm

add_path('pymbvi')
from pymbvi.models.mjp.autograd_partition_specific_models import SimpleGeneExpression
from pymbvi.util import num_derivative, autograd_jacobian

# torch.manual_seed(2007301620)

load_path = os.path.dirname(os.path.realpath(__file__)) + '/data/learn_linear_ode_train.pt'
data = torch.load(load_path)
model_dict = data['model_state_dict']
loss_history = data['loss_history']
print(len(loss_history))

# get simulation model
pre, post, rates = sm.get_standard_model("simple_gene_expression")

# prepare initial conditions
initial = np.array([0.0, 1.0, 0.0, 0.0])
tspan = np.array([0.0, 3e3])

# # set up gene expression model
moment_initial = np.zeros(9)
moment_initial[0:3] = initial[1:4]
model = SimpleGeneExpression(moment_initial, np.log(np.array(rates)), tspan)


class LinearODE(torch.nn.Module):

    def __init__(self, A, b):
        super(LinearODE, self).__init__()
        self.A = torch.nn.Parameter(A.clone())
        self.b = torch.nn.Parameter(b.clone())

    def forward(self, time, state):
        dydt = self.A @ state + self.b
        return(dydt)


# get A for linear gene expression model
rates = torch.tensor(rates).log()
moment_initial = torch.tensor(moment_initial)
def fun(state):
    tmp = model.forward_torch(0.0, state, torch.zeros(rates.shape), rates)
    return(tmp)
A_true = autograd_jacobian(fun, moment_initial)
b_true = fun(torch.zeros(moment_initial.shape))
model = LinearODE(A_true, b_true)

# compute true solution
t_eval = torch.arange(tspan[0], tspan[1], 20)
with torch.no_grad():
    sol = odeint(model, moment_initial, t_eval)

# evalute fitted model
model.load_state_dict(model_dict)
with torch.no_grad():
    sol_final = odeint(model, moment_initial, t_eval)
A_fit = model.A.clone().detach()
b_fit = model.b.clone().detach()

# get data
t_data = t_eval[0::15]
data = sol[0::15]

cmap = 'hot'
vmax = 0.2 #np.max(A_true.numpy())
vmin = -0.01 #np.min(A_true.numpy())
plt.subplot(1, 2, 1)
plt.imshow(A_true, cmap=cmap, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(A_fit, cmap=cmap, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.show()

# plot
for i in range(8):
    plt.subplot(3, 3, i+1)
    plt.plot(t_eval, sol[:, i], '-b')
    plt.plot(t_eval, sol_final[:, i], '-r')
    plt.plot(t_data, data[:, i], 'xk')
plt.show()
