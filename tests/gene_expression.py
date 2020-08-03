import numpy as np
import torch
import matplotlib.pyplot as plt
#from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint
from linear_memory.linear_memory import LinearMemory
import linear_memory.utils as ut

torch.manual_seed(2007301620)

# load meta data
load_path = '/Users/christian/Documents/Code/linear_memory/data/gene_expression/data.npz'
data = np.load(load_path)
rates = data['rates']
initial = data['initial']
num_samples = data['num_samples'] 
tspan = data['tspan']
t_plot = data['t_plot']
delta_t = data['delta_t']
t_obs = data['t_obs']
states_rre = data['states_rre']
num_times = len(t_plot)
num_dims = data['states_plot_0'].shape[-1]

# set up matrices
A_SS = torch.zeros((2, 2))
A_SS[0, 0] = -rates[5]
A_SS[1, 0] = rates[5]
A_SS[1, 1] = -2*rates[5]
A_SE = torch.zeros((2, 2))
A_SE[0, 0] = rates[4]
A_SE[1, 0] = rates[4]
A_SE[1, 1] = 2*rates[4]


class GeneExpression(LinearMemory):

    def forward_subnet(self, time, subnet):
        dydt = A_SS @ subnet
        return(dydt)


model = GeneExpression(2, 2, torch.tensor(rates), A_SE)

# comput solution
t_eval = torch.from_numpy(t_plot)
initial = torch.zeros(4)
#initial[0] = 100
#initial[1] = 100**2
with torch.no_grad():
    sol = odeint(model, initial, t_eval).detach()

# load trajectories
states = torch.zeros((num_samples, num_times, num_dims))
for i in range(num_samples):
    states[i] = torch.tensor(data['states_plot_{}'.format(i)])
#dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=2, shuffle=False)
# states for plotting
states_mean = torch.mean(states, axis=0)
states_mean_sq = torch.std(states**2, axis=0)

# set up optimzier
params = model.parameters()
#optimizer = torch.optim.Adam(params, lr=1e-2)
optimizer = torch.optim.SGD(params, lr=1e-7)

def closure():
    optimizer.zero_grad()
    predict = odeint(model, initial, t_eval)
    loss = ((predict[:, 0]-states_mean[:, 3])**2).sum()/len(t_eval)
    loss.backward()
    return loss

# perform optimization
max_epochs = 10
msg = "Loss in epoch {0} is {1}"
for epoch in range(max_epochs):
    optimizer.zero_grad()
    # solve forward and update loss
    predict = odeint(model, initial, t_eval)
    loss = ((predict[:, 0]-states_mean[:, 3])**2).sum()/len(t_eval)
    #loss2 = ((predict[:, 1].abs().sqrt()-states_mean_sq[:, 3].sqrt())**2).sum()/len(t_eval)
    #loss = loss1#+loss2
    # optimization step
    loss.backward()
    optimizer.step()
    # with torch.no_grad():
    #     loss = closure()
    print(msg.format(epoch+1, loss.item()))


with torch.no_grad():  
    sol_final = odeint(model, initial, t_eval)


# plot
plt.subplot(4, 2, 1) 
plt.plot(t_plot, states_mean[:, 1], ':r')
plt.plot(t_plot, states_rre[:, 1], '-r')
plt.subplot(4, 2, 2)
plt.plot(t_plot, states_mean_sq[:, 1], ':r')
plt.subplot(4, 2, 3)
plt.plot(t_plot, states_rre[:, 2], '-r')
plt.plot(t_plot, states_mean[:, 2], ':r')
plt.subplot(4, 2, 4)
plt.plot(t_plot, states_mean_sq[:, 2], ':r')
plt.subplot(4, 2, 5)
plt.plot(t_plot, states_mean[:, 3], ':r')
plt.plot(t_plot, states_rre[:, 3], '-r')
#plt.plot(t_eval, sol[:, 0], '-b')
plt.plot(t_eval, sol_final[:, 0], '-k')
plt.subplot(4, 2, 6)
plt.plot(t_plot, states_mean_sq[:, 3], ':r')
#plt.plot(t_eval, sol[:, 1], '-b')
plt.plot(t_eval, sol_final[:, 1], '-k')
plt.subplot(4, 2, 7)
plt.plot(t_eval, sol[:, 2], '-b')
plt.plot(t_eval, sol_final[:, 2], '-k')
plt.subplot(4, 2, 8)
plt.plot(t_eval, sol[:, 3], '-b')
plt.plot(t_eval, sol_final[:, 3], '-k')
plt.show()