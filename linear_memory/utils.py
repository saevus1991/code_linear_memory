import numpy as np

def get_stats(trajectories):
    """
    extract mean and covariance from a gillespie2 results file
    """
    # compute mean and cov
    mean = np.mean(trajectories, axis=0)
    tmp = trajectories - np.expand_dims(mean, 0)
    cov = np.mean(np.expand_dims(tmp, -1) @ np.expand_dims(tmp, -2), axis=0)
    return(mean, cov)