import sys
import sys
import numpy as np 
import matplotlib.pyplot as plt

pyssa_path = '/Users/christian/Documents/Code/pyssa'
sys.path.append(pyssa_path)
import pyssa.util as ut

# load data
load_path = '/Users/christian/Documents/Code/linear_memory/data/gene_expression/data.npz'
data = np.load(load_path)
num_samples = data['num_samples']
tspan = data['tspan']
t_plot = data['t_plot']
delta_t = data['delta_t']
t_obs = data['t_obs']
states_rre = data['states_rre']
num_steps, num_species = data['states_plot_0'].shape

trajectories = np.zeros((num_samples, num_steps, num_species))
for i in range(num_samples):
    trajectories[i] = data['states_plot_{}'.format(i)]
states_ssa = data['states_plot_0']
mean_ssa, cov_ssa = ut.get_stats(trajectories)

plt.plot(t_plot, 100*mean_ssa[:, 1], '-k')
plt.plot(t_plot, mean_ssa[:, 2], '-b')
plt.plot(t_plot, mean_ssa[:, 3], '-r')
plt.show()