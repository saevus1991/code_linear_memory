import numpy as np
import torch
import sys
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

torch.manual_seed(2007301620)

# get simulation model
pre, post, rates = sm.get_standard_model("simple_gene_expression")

# prepare initial conditions
initial = np.array([0.0, 1.0, 0.0, 0.0])
tspan = np.array([0.0, 3e3])

# set up gene expression model
moment_initial = np.zeros(9)
moment_initial[0:3] = initial[1:4]
model = SimpleGeneExpression(moment_initial, np.log(np.array(rates)), tspan)


class LinearODE(torch.nn.Module):

    def __init__(self, A, b):
        super(LinearODE, self).__init__()
        self.A = torch.nn.Parameter(A)
        self.b = torch.nn.Parameter(b)

    def forward(self, time, state):
        dydt = self.A @ state + self.b
        return(dydt)


# get A for linear gene expression model
rates = torch.tensor(rates).log()
moment_initial = torch.tensor(moment_initial)
def fun(state):
    tmp = model.forward_torch(0.0, state, torch.zeros(rates.shape), rates)
    return(tmp)
A = autograd_jacobian(fun, moment_initial)
b = fun(torch.zeros(moment_initial.shape))
model = LinearODE(A, b)

# compute true solution
t_eval = torch.arange(tspan[0], tspan[1], 20)
with torch.no_grad():
    sol = odeint(model, moment_initial, t_eval)

# reset model parameters
model.A = torch.nn.Parameter(torch.zeros(model.A.shape))
model.b = torch.nn.Parameter(torch.zeros(model.b.shape))

# get data
t_data = t_eval[0::15]
data = sol[0::15]

# optimizer 
params = model.parameters()
optimizer = torch.optim.SGD(params, lr=2e-10, momentum=0.99)


# # fit
# epochs = 0
# msg = 'Loss in epoch {0} is {1}'
# for i in range(epochs):
#     optimizer.zero_grad()
#     predict = odeint(model, moment_initial, t_data)
#     #loss = torch.sum((data-predict)**2/torch.sum(data**2, axis=0, keepdim=True))
#     loss = torch.sum(((data-predict)/(data+1))**2)
#     loss.backward()
#     optimizer.step()
#     print(msg.format(i, loss.item()))
# with torch.no_grad():
#     sol_final = odeint(model, moment_initial, t_eval)


# # plot
# for i in range(8):
#     plt.subplot(3, 3, i+1)
#     plt.plot(t_eval, sol[:, i], '-b')
#     plt.plot(t_eval, sol_final[:, i], '-r')
#     plt.plot(t_data, data[:, i], 'xk')
# plt.show()
