import numpy as np
import abstract_qae
from scipy.special import erfinv

def log_likelihood(grover_depths, eps, shots, zeros):
    eps_use = eps/3
    ones = shots - zeros
    thetas = np.arange(eps_use, np.pi/2, eps_use)
    ll = np.zeros_like(thetas)
    for j, grover_depth in enumerate(grover_depths):
        D = (2*grover_depth) + 1
        ll += zeros[j]*np.log(np.sin(D*thetas)**2)
        ll += ones[j]*np.log(np.cos(D*thetas)**2)
    return thetas, ll

def max_log_likelihood(grover, grover_depths, eps, shots, zeros):
    thetas, ll = log_likelihood(grover_depths, eps, shots, zeros)
    idx = np.argmax(ll)
    theta = thetas[idx]
    return theta

def qae(probability, max_grover_depth, eps, delta, shot_scale=1.0):
    shots = calculate_Nshot(max_grover_depth, eps, delta)
    shots = int(np.ceil(shots*shot_scale))
    grover_depths = calc_depths(max_grover_depth)
    total_shots, total_calls = 0, 0
    zeros = []
    for j, grover_depth in enumerate(grover_depths):
        D = (2*grover_depth) + 1
        total_shots += shots
        total_calls += D*shots
        zerosd = abstract_qae.grover_from_probability(probability, \
            grover_depth, shots)
        zeros.append(zerosd)
    zeros = np.array(zeros)
    shots_array = np.array([shots]*len(grover_depths))
    theta_est = max_log_likelihood(probability, grover_depths, eps, \
        shots_array, zeros)
    probability_est = abstract_qae.probability_from_theta(theta_est)
    return probability_est, total_shots, total_calls, shots

def qae2(probability, max_grover_depth, eps, delta_in):
    # may need to adjust delta?
    grover_depths = calc_depths(max_grover_depth)
    delta = 1-((1-delta_in)**(1/len(grover_depths)))
    Ds = (2*grover_depths)+1
    shots_array = []
    for j, grover_depth in enumerate(grover_depths):
        if grover_depth == 0:
            eps_tmp = np.pi/(4*Ds[j+1])
            shots = np.log(2/delta)/(2*(eps_tmp**2))
            shots = int(np.ceil(shots))
            shots_array.append(shots)
        elif grover_depth == grover_depths[-1]:
            eps_tmp = eps*Ds[j]
            shots = np.log(2/delta)/(2*(eps_tmp**2))
            shots = int(np.ceil(shots))
            shots_array.append(shots)
        else:
            eps_tmp = ( np.pi/(4*Ds[j+1]) )*Ds[j]
            shots = np.log(2/delta)/(2*(eps_tmp**2))
            shots = int(np.ceil(shots))
            shots_array.append(shots)
    shots_array = np.array(shots_array)

    total_shots, total_calls = 0, 0
    zeros = []
    for j, grover_depth in enumerate(grover_depths):
        D = (2*grover_depth) + 1
        total_shots += shots_array[j]
        total_calls += D*shots_array[j]
        zerosd = abstract_qae.grover_from_probability(probability, \
            grover_depth, shots_array[j])
        zeros.append(zerosd)
    zeros = np.array(zeros)
    theta_est = max_log_likelihood(probability, grover_depths, eps, \
        shots_array, zeros)
    probability_est = abstract_qae.probability_from_theta(theta_est)
    return probability_est, total_shots, total_calls, shots_array

def qae3(probability, max_grover_depth, eps, delta):
    # tries to reuse previous shots to disambiguate the final step
    shots = calculate_Nshot(max_grover_depth, eps, delta)
    grover_depths = calc_depths(max_grover_depth)
    total_shots, total_calls = 0, 0
    zeros = []
    for j, grover_depth in enumerate(grover_depths):
        D = (2*grover_depth) + 1
        total_shots += shots
        total_calls += D*shots
        zerosd = abstract_qae.grover_from_probability(probability, \
            grover_depth, shots)
        zeros.append(zerosd)
    zeros = np.array(zeros)
    shots_array = np.array([shots]*len(grover_depths))
    theta_est = max_log_likelihood(probability, grover_depths, eps, \
        shots_array, zeros)
    probability_est = abstract_qae.probability_from_theta(theta_est)

    D = (2*grover_depths[-1])+1
    lines = np.array([j*np.pi/(2*D)])
    idx = np.argmin(np.abs(theta_est - lines))
    closest_line = lines[idx]
    distance = np.abs(theta_est - closest_line)
    direction = np.sign(theta_est - closest_line)
    if distance < eps:
        probability_est_out = probability_est
        req_extra, idx = 0, 0
    else:
        theta_est_other = closest_line - (direction*distance)
        ps, p_others, pdiffs, reqs, req_calls = [], [], [], [], []
        for j, grover_depth in enumerate(grover_depths):
            D = (2*grover_depth) + 1
            p, p_other = np.sin(theta_est*D)**2, np.sin(theta_est_other*D)**2
            pdiff = np.abs(p-p_other)
            req = np.log(2/delta)/(2*(((pdiff))**2))
            req_call = req*D
            ps.append(p), p_others.append(p_other)
            pdiffs.append(pdiff), reqs.append(req), req_calls.append(req_call)
        idx = np.argmin(req_calls)
        req_extra = reqs[idx] - shots_array[idx]
        if req_extra <= 0:
            zeros_disambiguate = zeros[idx]
            shots_disambiguate = shots_array[idx]
        else:
            req_extra = int(np.ceil(req_extra))
            new_zeros = abstract_qae.grover_from_probability(probability, \
            grover_depths[idx], req_extra)
            total_shots += req_extra
            total_calls += req_extra*((2*grover_depths[idx])+1)
            zeros_disambiguate = zeros[idx] + new_zeros
            shots_disambiguate = shots_array[idx] + req_extra
        p_disambiguate = zeros_disambiguate/shots_disambiguate
        if np.abs(p_disambiguate-ps[idx]) < \
            np.abs(p_disambiguate-p_others[idx]):
            probability_est_out = probability_est
        else:
            probability_est_other = np.sin(theta_est_other)**2
            probability_est_out = probability_est_other

    return probability_est_out, total_shots, total_calls, shots, req_extra, \
        grover_depths[idx]

def qae4(probability, max_grover_depth, eps, delta, jittigate=False):
    shots = calculate_Nshot(max_grover_depth, eps, delta)
    grover_depths_nojitter = calc_depths(max_grover_depth)
    total_shots, total_calls = 0, 0
    zeros = []
    jittigate_now = False
    shots_array = []
    grover_depths = []
    for j, grover_depth in enumerate(grover_depths_nojitter):
        #if grover_depth == grover_depths_nojitter[-1]:
        if grover_depth > 0:
            jittigate_now = True
        if not jittigate_now:
            grover_depths += [grover_depth]
            shots_array += [shots]
            continue
        gdspread = np.log(2*grover_depth)
        if grover_depth - gdspread <= grover_depths_nojitter[j-1]:
            grover_depths += [grover_depth]
            shots_array += [shots]
            continue
        gdcurrmin = np.max((grover_depths_nojitter[j-1]+1, \
            int(np.round(grover_depth-gdspread))))
        if grover_depth == grover_depths_nojitter[-1]:
            gdcurrmax = grover_depths_nojitter[-1]
        else:
            gdcurrmax = np.min((grover_depths_nojitter[j+1]-1, \
                int(np.round(grover_depth+gdspread))))          
        grover_depths_current = list(range(gdcurrmin, gdcurrmax+1))
        shots_curr = int(np.ceil(shots/len(grover_depths_current)))
        grover_depths += grover_depths_current
        shots_array += [shots_curr]*len(grover_depths_current)

    #print(grover_depths)
    #print(shots_array)
    grover_depths = np.array(grover_depths)
    shots_array = np.array(shots_array)
    for j, grover_depth in enumerate(grover_depths):
        D = (2*grover_depth) + 1
        total_shots += shots_array[j]
        total_calls += D*shots_array[j]
        zerosd = abstract_qae.grover_from_probability(probability, \
            grover_depth, shots_array[j])
        zeros.append(zerosd)
    zeros = np.array(zeros)
    theta_est = max_log_likelihood(probability, grover_depths, eps, \
        shots_array, zeros)
    probability_est = abstract_qae.probability_from_theta(theta_est)
    return probability_est, total_shots, total_calls, shots

def calc_depths(maxdepth):
    if maxdepth == 0:
        return np.array([0])
    elif maxdepth == 1:
        return np.array([1])
    kmaxapprox = np.log2(maxdepth)
    kmaxs = (int(np.ceil(kmaxapprox)), int(np.floor(kmaxapprox)))
    if kmaxs[1] == 0:
        base = maxdepth**(1/kmaxs[0])
    else:
        base = (maxdepth**(1/kmaxs[0]), maxdepth**(1/kmaxs[1]))
    ind = np.argmin((np.abs(base[0] - 2), np.abs(base[1]-2)))
    base, kmax = base[ind], kmaxs[ind]

    mks = np.array([0] + [int(np.round(base**k)) for k in range(kmax+1)], \
        dtype=int)

    return mks

def _Smoment(maxdepth, power):
    mks = calc_depths(maxdepth)
    return np.sum( ( ( (2*mks) + 1 )**power ) )**(1/power)

def S1(maxdepth, depth_jitter=None):
    return _Smoment(maxdepth, 1)

def S2(maxdepth, depth_jitter=None):
    return _Smoment(maxdepth, 2)

def calculate_Nshot(maxdepth, eps, delta):
    num = erfinv(1-delta)**2
    denom = 2 * ((S2(maxdepth)*eps)**2)
    return int(np.ceil(num/denom))

def calculate_eps(maxdepth, Nshot, delta):
    num = erfinv(1-delta)
    denom = np.sqrt(2) * S2(maxdepth) * np.sqrt(Nshot)
    return num/denom