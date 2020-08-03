import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

pyssa_path = '/Users/christian/Documents/Code/pyssa'
sys.path.append(pyssa_path)
import pyssa.util as ut

# load data from file
load_path = '/Users/christian/Documents/Code/linear_memory/data/stochastic_toggle_switch/data.npz'
data = np.load(load_path)
num_samples = data['num_samples']
tspan = data['tspan']
t_plot = data['t_plot']
delta_t = data['delta_t']
t_obs = data['t_obs']
states_rre = data['states_rre']
num_steps, num_species = data['states_ssa_0'].shape

trajectories = np.zeros((num_samples, num_steps, num_species))
for i in range(num_samples):
    trajectories[i] = data['states_ssa_{}'.format(i)]
states_ssa = data['states_ssa_0']
mean_ssa, cov_ssa = ut.get_stats(trajectories)

# get torch data
data = torch.from_numpy(trajectories)
time = torch.from_numpy(t_plot)


# # plot
# colors = ['r', 'b', 'g']
# plt.subplot(3, 1, 1)
# for i in range(2):
#     ind = 1+4*i
#     plt.plot(t_plot/3600, states_ssa[:, ind], ':', color=colors[i])
#     plt.plot(t_plot/3600, states_rre[:, ind], '.', color=colors[i])
#     plt.plot(t_plot/3600, mean_ssa[:, ind], '-', color=colors[i])
# plt.subplot(3, 1, 2)
# for i in range(2):
#     ind = 2+4*i
#     plt.plot(t_plot/3600, states_ssa[:, ind], ':', color=colors[i])
#     plt.plot(t_plot/3600, states_rre[:, ind], '.', color=colors[i])
#     plt.plot(t_plot/3600, mean_ssa[:, ind], '-', color=colors[i])
# plt.subplot(3, 1, 3)
# for i in range(2):
#     ind = 3+4*i
#     plt.plot(t_plot/3600, states_ssa[:, ind], ':', color=colors[i])
#     plt.plot(t_plot/3600, states_rre[:, ind], '.', color=colors[i])
#     plt.plot(t_plot/3600, mean_ssa[:, ind], '-', color=colors[i])
# plt.show()