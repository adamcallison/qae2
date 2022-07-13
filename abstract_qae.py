import numpy as np
from scipy.stats import binom

def qae(probability, grover_depth, eps, delta_in):

    delta = 1-((1-delta_in)**(1/3))
    
    D = (2*grover_depth) + 1

    
    # computing the required shots at grover depth
    eps_tmp = eps*D
    shots = np.log(2/delta)/(2*(eps_tmp**2))
    shots = int(np.ceil(shots))
    
    # computing the required precision (and then shots) at zero depth
    eps_0 = np.pi/(4*((2*grover_depth)+1))
    shots_0 = np.log(2/delta)/(2*(eps_0**2))
    shots_0 = int(np.ceil(shots_0))

    # computing a rough estimate to within eps_0
    results_0 = grover_from_probability(probability, 0, shots_0)
    total_shots = shots_0
    total_calls = shots_0
    theta_rough_est = np.arcsin(np.sqrt(results_0/shots_0))
    
    # nval and sign tell us which width-(pi/2D) region the true angle is in
    nval = np.round( theta_rough_est/(np.pi/(2*D)) )
    sign = np.sign( theta_rough_est - (nval*np.pi/(2*D)) )
    
    # if true angle is close to 0 or pi/2, no ambiguity in result
    disambiguate = True
    if (np.abs(nval - 0) < 0.1) or (np.abs(nval - D) < 0.1):
        disambiguate = False
    # if rough estimate is close to middle of region, no ambiguity in result
    elif np.abs( (nval*(np.pi/(2*D))) - theta_rough_est ) > \
        np.pi/(4*((2*grover_depth)+1)):
        disambiguate = False
    
        
    # now do the shots at grover depth for precise estimate
    results_1 = grover_from_probability(probability, grover_depth, shots)
    total_shots += shots
    total_calls += shots*D

    # combine both results to get a precise estimate
    amp_theta_est_piece_orig = np.arcsin(np.sqrt(results_1/shots))
    if nval%2 == 1:
        amp_theta_est_piece = (np.pi/2) - amp_theta_est_piece_orig
    else:
        amp_theta_est_piece = amp_theta_est_piece_orig
    theta_est1 = (sign*(amp_theta_est_piece/D) )+ (nval*np.pi/(2*D))
    
    # compute the other estimate, in case the rough estimate put us the wrong 
    # side of the axis
    if disambiguate:
        theta_est2 = ((-1.0*sign)*(amp_theta_est_piece/D) )+ (nval*np.pi/(2*D))
    else:
        theta_est2 = theta_est1
        
    probability_est1 = np.sin( theta_est1 )**2
    if np.abs(theta_est1 - theta_est2) <= eps:
        # if the two estimates are close, just use the first one (does this 
        # lead to slightly increased error??)
        probability_est = probability_est1
        shots_2 = 0.0
        pdiff = 0.0
        jbest = 0
    else:
        # else we need to look at a different depth to disambiguate
        probability_est2 = np.sin( theta_est2 )**2

        ps = []
        p2s = []
        pdiffs = []
        shots_2s = []
        new_callss = []
        
        # find the smaller grover_depth j that will disambiguate with the least
        # total calls
        for j in range(grover_depth):
            J = (2*j)+1
            p = np.sin(J*theta_est1)**2
            p2 = np.sin(J*theta_est2)**2
            ps.append(p)
            p2s.append(p2)
            pdiff = np.abs(p2-p)
            pdiffs.append(pdiff)
            shots_2 = np.log(2/delta)/(2*((pdiff/2)**2))
            shots_2s.append(shots_2)
            new_calls = shots_2*J
            new_callss.append(new_calls) 
        jbest = np.argmin(new_callss)
        pdiff = pdiffs[jbest]
        shots_2 = shots_2s[jbest]
        shots_2 = int(np.ceil(shots_2))

        # compute estimate as tightly as we need (need to somehow make sure the
        # total number of shots is sensibly bounded)
        results_2 = grover_from_probability(probability, jbest, shots_2)
        total_shots += shots_2
        total_calls += shots_2*((2*jbest)+1)
        p3 = results_2/shots_2

        # then pick the closest of the two estimates from the grover_depth
        # circuit
        if np.abs(p3 - ps[jbest]) < np.abs(p3 - p2s[jbest]):
            probability_est = probability_est1
        else:
            probability_est = probability_est2
    
    extra_stats = {'shots':shots, 'shots_rough':shots_0, \
        'shots_disambiguate':shots_2, 'depth_disambiguate':jbest}
    
    return probability_est, total_shots, total_calls, extra_stats

def theta_from_probability(probability):
    return np.arcsin(np.sqrt(probability))

def probability_from_theta(theta):
    return np.sin(theta)**2

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

def measure_from_theta(theta, shots):
    probability = probability_from_theta(theta)
    return measure_from_probability(probability, shots)

def grover_from_theta(theta, grover_depth, shots):
    amp_theta = amplitified_theta(theta, grover_depth)
    return measure_from_theta(amp_theta, shots)

def grover_from_probability(probability, grover_depth, shots):
    amp_probability = amplified_probability(probability, grover_depth)
    return measure_from_probability(amp_probability, shots)

