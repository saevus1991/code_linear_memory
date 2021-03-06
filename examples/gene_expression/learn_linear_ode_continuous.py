import numpy as np
import torch
import os
import matplotlib.pyplot as plt
#from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint
from linear_memory.linear_memory import LinearMemory
import linear_memory.utils as ut
from import_utils import add_path
from scipy.interpolate import interp1d

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

def searchsorted(vector, elements):
    ind = np.searchsorted(vector.detach().numpy(), elements.detach().numpy())
    return(torch.tensor(ind))

class Linfun(torch.autograd.Function):
    """
    A step function over the interval [0, 1]
    """

    @staticmethod
    def forward(ctx, time, times, vals):
        if time < times[0]:
            output = vals[0]
            if time.requires_grad:
                alpha = torch.zeros(output.shape)
                ctx.save_for_backward(alpha)
        elif time >= times[-1]:
            output = vals[-1]
            if time.requires_grad:
                alpha = torch.zeros(output.shape)
                ctx.save_for_backward(alpha)
        else:
            ind = searchsorted(times, time)
            alpha = (vals[ind]-vals[ind-1])/(times[ind]-times[ind-1])
            output = vals[ind-1] + alpha*(time-times[ind-1])
            if time.requires_grad:
                ctx.save_for_backward(alpha)
        return(output)

    @staticmethod
    def backward(ctx, grad_output):
        grad_time = (alpha * grad_output).sum()
        return(grad_time, None, None)


class LinearODE(torch.nn.Module):

    def __init__(self, A, b):
        super(LinearODE, self).__init__()
        self.A = torch.nn.Parameter(A)
        self.b = torch.nn.Parameter(b)

    def forward(self, time, state):
        dydt = self.A @ state + self.b
        return(dydt)


class LinearODECnt(torch.nn.Module):

    def __init__(self, A, b, t_grid, data):
        super(LinearODECnt, self).__init__()
        self.A = torch.nn.Parameter(A)
        self.b = torch.nn.Parameter(b)
        self.t_grid = t_grid
        self.data = data
        self.interp = interp1d(t_grid, data, axis=0, bounds_error=False, fill_value=(data[0], data[-1]))
        self.alpha = 1.0/(len(b)*(t_grid[-1]-t_grid[0]))

    def forward(self, time, state):
        dydt = self.A @ state[:-1] + self.b
        state_ref = self.compute_reference(time.detach())
        tmp = self.A.sum() + self.b.sum()
        dloss = self.alpha*torch.sum(((state[:-1]-state_ref)/(0.1+state_ref))**2, axis=0, keepdim=True) 
        return(torch.cat([dydt, dloss]))

    def compute_reference(self, time):
        #state_ref = torch.from_numpy(self.interp(time))
        state_ref = Linfun.apply(time, self.t_grid, self.data) 
        return(state_ref)



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

# get data
t_data = t_eval[0::15]
data = sol[0::15]

# set up model with continuous error
moment_initial = torch.cat([moment_initial, torch.tensor([0.0])])
model = LinearODECnt(torch.zeros(model.A.shape), torch.zeros(model.b.shape), t_data, data)

# reset model parameters
model.A = torch.nn.Parameter(1e-6*torch.randn(model.A.shape))
model.b = torch.nn.Parameter(1e-6*torch.randn(model.b.shape))

# optimizer 
params = model.parameters()
#optimizer = torch.optim.SGD(params, lr=1e-9)
optimizer = torch.optim.LBFGS(params, lr=1, line_search_fn='strong_wolfe')

# def loss_fn(predict, data):
#     loss = torch.sum(((predict-data)/(data+1))**2)
#     return(loss)

def l1(model):
    loss = 0.0
    for p in model.parameters():
        loss += torch.abs(p).sum()
    return(loss)

def closure():
    if torch.is_grad_enabled():
        optimizer.zero_grad()
    predict = odeint(model, moment_initial, t_data)
    loss = 10*predict[-1, -1] + 10*l1(model)
    if loss.requires_grad:
        loss.backward()
    return(loss)

# fit
max_epoch = 100
loss_history = []
save_path = os.path.dirname(os.path.realpath(__file__)) + '/data/learn_linear_ode_continuous_train.pt'
msg = 'Loss in epoch {0} is {1}'
for epoch in range(max_epoch):
    loss = optimizer.step(closure)
    loss_history.append(loss.item())
    with torch.no_grad():
        loss2 = 10*l1(model)
    print(msg.format(epoch, loss.item()))
    print('Parameter loss is {}'.format(loss2))
    # save
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_history': torch.tensor(loss_history)}, save_path)
with torch.no_grad():
    sol_final = odeint(model, moment_initial, t_eval)

print(sol_final[-1])

# # plot
# for i in range(8):
#     plt.subplot(3, 3, i+1)
#     plt.plot(t_eval, sol[:, i], '-b')
#     plt.plot(t_eval, sol_final[:, i], '-r')
#     plt.plot(t_data, data[:, i], 'xk')
# plt.show()
