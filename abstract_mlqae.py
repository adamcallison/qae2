import numpy as np
from scipy.stats import binom
import mlqae

def qae(probability, max_grover_depth, eps, delta, Nshot=None, \
    shot_multiplier=None, jittigate=False):
    return mlqae.qae('abstract', (probability,), qae_zerocounts, \
        max_grover_depth, eps, delta, Nshot=Nshot, \
        shot_multiplier=shot_multiplier, jittigate=jittigate)

def qae_zerocounts(arg_tuple, grover_depths, shots_array, **kwargs):
    probability = arg_tuple[0]
    total_shots, total_calls = 0, 0
    zeros = []
    for j, grover_depth in enumerate(grover_depths):
        D = (2*grover_depth) + 1
        total_shots += shots_array[j]
        total_calls += D*shots_array[j]
        zerosd = grover_from_probability(probability, \
            grover_depth, shots_array[j])
        zeros.append(zerosd)
    zeros = np.array(zeros)
    return zeros, total_shots, total_calls

def theta_from_probability(probability):
    return np.arcsin(np.sqrt(probability))

def amplitified_theta(theta, grover_depth):
    D = (2*grover_depth)+1
    return D*theta

def amplified_probability(probability, grover_depth):
    theta = theta_from_probability(probability)
    amp_theta = amplitified_theta(theta, grover_depth)
    return np.sin(amp_theta)**2

def measure_from_probability(probability, shots):
    p_0, p_1 = 1-probability, probability
    zeros = binom.rvs(shots, p_1)
    return zeros

def grover_from_probability(probability, grover_depth, shots):
    amp_probability = amplified_probability(probability, grover_depth)
    return measure_from_probability(amp_probability, shots)
