import numpy as np
from scipy.stats import binom
from mlqae import calc_depths, calculate_Nshot, max_log_likelihood, \
    probability_from_theta

def qae(probability, max_grover_depth, eps, delta, Nshot=None, \
    jittigate=False):
    grover_depths, shot_scales = calc_depths(max_grover_depth, \
        jittigate=jittigate)
    if Nshot is None:
        shots = calculate_Nshot(max_grover_depth, eps, delta, \
            jittigate=jittigate)
    else:
        shots = Nshot
    shots_array = np.array([int(np.ceil(x)) for x in (shots*shot_scales)])
    total_shots, total_calls = 0, 0
    zeros = []
    grover_depths = np.array(grover_depths)
    shots_array = np.array(shots_array)
    for j, grover_depth in enumerate(grover_depths):
        D = (2*grover_depth) + 1
        total_shots += shots_array[j]
        total_calls += D*shots_array[j]
        zerosd = grover_from_probability(probability, \
            grover_depth, shots_array[j])
        zeros.append(zerosd)
    zeros = np.array(zeros)
    theta_est = max_log_likelihood(grover_depths, eps, shots_array, zeros)
    probability_est = probability_from_theta(theta_est)
    return probability_est, total_shots, total_calls, shots

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
