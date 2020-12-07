import sys
import os
from pathlib import Path
import numpy as np 
import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint

workdir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(workdir))
from import_utils import add_path

add_path('pyssa')
#pyssa_path = '/Users/christian/Documents/Code/pyssa'
#sys.path.append(pyssa_path)
import pyssa.util as ut

torch.set_default_dtype(torch.float64)
torch.manual_seed(2008181715)

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

trajectories = np.zeros((num_samples, num_steps, num_species))
for i in range(num_samples):
    trajectories[i] = data['states_plot_{}'.format(i)]
trajectories = trajectories[:, :, 1:]
num_species = num_species-1
states_ssa = data['states_plot_0']

def get_moments(trajectories):
    mean_ssa, cov_ssa = ut.get_stats(trajectories)
    states_mean = np.concatenate([mean_ssa, np.stack([cov_ssa[:, i, j] for i in range(3) for j in range(i, 3)]).T], axis=1)
    return(states_mean)


# get noise estimates via bootstrapping
batch_size = 100
num_iter = 1000
states_mean = np.zeros((num_steps, int(num_species*(num_species+3)/2)))
states_err = np.zeros((num_steps, int(num_species*(num_species+3)/2)))
np.random.seed(2008181434)
for i in range(num_iter):
    samples = np.random.randint(num_samples, size=(batch_size,))
    tmp = trajectories[samples]
    mean_tmp = get_moments(tmp)
    states_mean += mean_tmp/num_iter
np.random.seed(2008181434)
for i in range(num_iter):
    samples = np.random.randint(num_samples, size=(batch_size,))
    tmp = trajectories[samples]
    mean_tmp = get_moments(tmp)
    states_err += (mean_tmp-states_mean)**2/num_iter
states_err = np.sqrt(states_err)

# set up  model 
class LinearODE(torch.nn.Module):

    def __init__(self, A, b):
        super(LinearODE, self).__init__()
        self.A = torch.nn.Parameter(A.clone())
        self.b = torch.nn.Parameter(b.clone())

    def forward(self, time, state):
        dydt = self.A @ state + self.b
        return(dydt)

# set up model
model = LinearODE(torch.zeros(A_true.shape), torch.zeros(b_true.shape))

# optimizer 
params = model.parameters()
optimizer = torch.optim.Adam(params, lr=1e-4, amsgrad=True)
#optimizer = torch.optim.SGD(params, lr=1e-11, momentum=0.9)

#set up dataload
dataloader = torch.utils.data.DataLoader(trajectories, batch_size=batch_size, shuffle=True)

def l1(model):
    loss = 0.0
    for p in model.parameters():
        loss += torch.abs(p).sum()
    return(loss)

def loss_fn(model, data):
    predict = odeint(model, moment_initial, torch.from_numpy(t_plot))
    empirical = torch.from_numpy(get_moments(data.numpy()))
    loss = torch.sum(((predict-empirical)/torch.from_numpy(1e-2+states_err))**2)/batch_size
    return(loss)

def closure():
    if torch.is_grad_enabled():
        optimizer.zero_grad()
    loss = loss_fn(model, data) #+ 10*l1(model)
    if loss.requires_grad:
        try:
            loss.backward()
        except:
            print("Error during backpropgation")
    return(loss)


# perform training
max_epoch = 200
loss_history = []
save_path = os.path.dirname(os.path.realpath(__file__)) + '/learn_linear_ode_minibatch.pt'
msg = 'Loss in epoch {0} is {1}'
for epoch in range(max_epoch):
    running_loss = 0.0
    for data in dataloader:
        loss = optimizer.step(closure)
        running_loss += loss.item()*(batch_size/num_samples)
        # with torch.no_grad():
        #     test = l1(model)
        print(loss.item())
        # print(test.item())
    loss_history.append(running_loss)
    print(msg.format(epoch, running_loss))
    # save
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_history': torch.tensor(loss_history),
                'states_mean': torch.from_numpy(states_mean),
                'states_err': torch.from_numpy(states_err)}, save_path)
with torch.no_grad():
    sol_final = odeint(model, moment_initial, torch.from_numpy(t_plot))
print(loss_history)

# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     plt.plot(t_plot, states_mean[:, i], '-g')
#     #plt.plot(t_plot, states_mb[:, i], '-r')
#     plt.plot(t_plot, sol_final[:, i], '-r')
#     plt.fill_between(t_plot, states_mean[:, i] - states_err[:, i], states_mean[:, i] + states_err[:, i], color='g', alpha=0.2)
# plt.show()
