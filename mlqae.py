import numpy as np
from scipy.special import erfinv, erf
import scipy.interpolate as spi

def probability_from_theta(theta):
    return np.sin(theta)**2

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

def max_log_likelihood2(grover_depths, eps, shots, zeros):

    tmp = ((2*grover_depths) + 1)**2
    tmp = tmp*shots
    tmp = np.sum( tmp )**(1/2)
    denom = np.sqrt(2) * tmp
    delta  = 1 - erf(denom*eps)

    eps_use = eps/3
    ones = shots - zeros
    for j, grover_depth in enumerate(grover_depths):

        j_use = np.max((2, j))
        num = erfinv(1-delta)
        tmp = ((2*grover_depths[:j_use+1]) + 1)**2
        tmp = tmp*shots[:j_use+1]
        tmp = np.sum( tmp )**(1/2)
        denom = np.sqrt(2) * tmp
        eps_curr = np.max( ((num/denom)/10, eps_use) )

        D = (2*grover_depth) + 1
 
        if j == 0:
            thetas_curr = np.arange(eps_use, (np.pi/2)-eps_use+eps_curr, \
                eps_curr)
            ll = zeros[j]*np.log(np.sin(D*thetas_curr)**2)
            ll += ones[j]*np.log(np.cos(D*thetas_curr)**2)
        elif j < 3:
            ll += zeros[j]*np.log(np.sin(D*thetas_curr)**2)
            ll += ones[j]*np.log(np.cos(D*thetas_curr)**2)
        else:
            thetas_old = thetas_curr
            ll_old = ll
            idx = np.argmax(ll)
            theta_curr = thetas_old[idx]
            theta_curr_min = np.max( (theta_curr - (eps_curr*50), eps_use) )
            theta_curr_max = np.min( (theta_curr + (eps_curr*50), np.pi/2) )
            thetas_curr = np.arange(theta_curr_min, theta_curr_max, eps_curr)
            ll = spi.interp1d(thetas_old, ll_old, kind='cubic', \
                fill_value="extrapolate")(thetas_curr)
            ll += zeros[j]*np.log(np.sin(D*thetas_curr)**2)
            ll += ones[j]*np.log(np.cos(D*thetas_curr)**2)

    idx = np.argmax(ll)
    theta = thetas_curr[idx]

    return theta

def max_log_likelihood(grover_depths, eps, shots, zeros):
    thetas, ll = log_likelihood(grover_depths, eps, shots, zeros)
    idx = np.argmax(ll)
    theta = thetas[idx]
    return theta

def calc_depths(maxdepth, jittigate=False):
    if maxdepth == 0:
        return np.array([0]), np.array([1.0])
    elif maxdepth == 1:
        return np.array([1]), np.array([1.0])
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

    if jittigate:
        mks, shot_scales = jitter_depths(mks)
    else:
        shot_scales = np.ones(len(mks))
    return mks, shot_scales

def jitter_depths(depths_nojitter, donly=False):
    if not donly:
        return _jitter_depths2(depths_nojitter)
    depths = list(depths_nojitter)[:-1]
    depth = depths_nojitter[-1]
    shot_scales = [1.0]*len(depths)
    dspread = int(np.round(np.log(2*depth)))
    new_depths = list(range(depth-dspread, depth+1))
    new_shot_scales = [1/len(new_depths)]*len(new_depths)
    depths = np.array(depths + new_depths)
    shot_scales = np.array(shot_scales + new_shot_scales)
    return depths, shot_scales

def _jitter_depths2(depths_nojitter):
    c = 2.0
    depths_nojitter = list(np.sort(depths_nojitter))
    depths_nojitter_rev = depths_nojitter[::-1]
    depths_rev, shot_scales_rev = [], []
    for j, depth_njr in enumerate(depths_nojitter_rev):
        if depth_njr == 0:
            jittigate_now = False
        elif j == 0:
            dspread = np.max((int(np.round(np.log(c*depth_njr))), 0))
            lowerd, upperd = depth_njr-dspread, depth_njr
            jittigate_now = lowerd > (depths_nojitter_rev[j+1]+1)
        elif j < len(depths_nojitter_rev) - 1:
            dspread = np.max((int(np.round(np.log(c*depth_njr))), 0))
            lowerd, upperd = depth_njr-dspread, depth_njr+dspread 
            jittigate_now = (lowerd > (depths_nojitter_rev[j+1]+1)) and \
                (upperd < (depths_rev[-1]-1))
        else:
            dspread = np.max((int(np.round(np.log(c*depth_njr))), 0))
            lowerd, upperd = np.max((depth_njr-dspread, 0)), depth_njr+dspread
            jittigate_now = upperd > (depths_rev[-1]-1)
    
        if jittigate_now:
            new_depthsr = list(range(upperd, (lowerd)-1, -1))
            new_shot_scalesr = [1/len(new_depthsr)]*len(new_depthsr)
            depths_rev = depths_rev + new_depthsr
            shot_scales_rev = shot_scales_rev + new_shot_scalesr
        else:
            depths_rev = depths_rev + [depth_njr]
            shot_scales_rev = shot_scales_rev + [1.0]

    depths = np.array(depths_rev[::-1])
    shot_scales = np.array(shot_scales_rev[::-1])

    return depths, shot_scales

def _jitter_depths(depths_nojitter):
    jittigate_now = False
    shot_scales = []
    depths = []
    for j, depth in enumerate(depths_nojitter):
        #if grover_depth == grover_depths_nojitter[-1]:
        if depth > 0:
            jittigate_now = True
        if not jittigate_now:
            depths += [depth]
            shot_scales += [1.0]
            continue
        dspread = np.log(2*depth)
        if depth - dspread <= depths_nojitter[j-1]:
            depths += [depth]
            shot_scales += [1.0]
            continue
        dcurrmin = np.max((depths_nojitter[j-1]+1, \
            int(np.round(depth-dspread))))
        if depth == depths_nojitter[-1]:
            dcurrmax = depths_nojitter[-1]
        else:
            dcurrmax = np.min((depths_nojitter[j+1]-1, \
                int(np.round(depth+dspread))))          
        depths_current = list(range(dcurrmin, dcurrmax+1))
        scale_curr = 1.0/len(depths_current)
        depths += depths_current
        shot_scales += [scale_curr]*len(depths_current)
    depths = np.array(depths)
    shot_scales = np.array(shot_scales)
    return depths, shot_scales

def _Smoment(maxdepth, power, jittigate=False):
    mks, shot_scales = calc_depths(maxdepth, jittigate=jittigate)
    tmp = ((2*mks) + 1)**power
    tmp = tmp*shot_scales
    return np.sum( tmp )**(1/power)

def S1(maxdepth, jittigate=False):
    return _Smoment(maxdepth, 1, jittigate=jittigate)

def S2(maxdepth, jittigate=False):
    return _Smoment(maxdepth, 2, jittigate=jittigate)

def calculate_Nshot(maxdepth, eps, delta, jittigate=False):
    num = erfinv(1-delta)**2
    denom = 2 * ((S2(maxdepth, jittigate=jittigate)*eps)**2)
    return int(np.ceil(num/denom))

def calculate_eps(maxdepth, Nshot, delta, jittigate=False):
    num = erfinv(1-delta)
    denom = np.sqrt(2) * S2(maxdepth, jittigate=jittigate) * np.sqrt(Nshot)
    return num/denom