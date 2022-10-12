import numpy as np 
from abstract_mlqae import calc_depths, calculate_Nshot, max_log_likelihood, \
    probability_from_theta
import circuit_util
from qiskit.providers.aer import AerSimulator
from qiskit import transpile

import qiskit_aer.noise as noise

def qae(A, O, measure_qubits, results_to_good_count_func, max_grover_depth, \
    eps, delta, Nshot=None, shot_multipler=None, jittigate=False, \
    noise_rate=None, kappa_params=None, compile_to=None):
    grover_depths, shot_scales = calc_depths(max_grover_depth, \
        jittigate=jittigate)
    if Nshot is None:
        shots = calculate_Nshot(max_grover_depth, eps, delta, \
            jittigate=jittigate)
    else:
        shots = Nshot
    if not shot_multipler == None:
        shots = int(np.ceil(shot_multipler*shots))
    shots_array = np.array([int(np.ceil(x)) for x in (shots*shot_scales)])
    total_shots, total_calls = 0, 0
    zeros = []
    grover_depths = np.array(grover_depths)
    shots_array = np.array(shots_array)
    circuits = circuit_util.qae_circuits(A, O, grover_depths, measure_qubits, \
        compile_to=compile_to)

    simulator = AerSimulator()
    if not (noise_rate is None):
        error = noise.depolarizing_error(noise_rate, 2)
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['cx'])
        job = simulator.run(circuits, shots=shots, noise_model=noise_model)
    else:
        job = simulator.run(circuits, shots=shots)

    for j, grover_depth in enumerate(grover_depths):
        D = (2*grover_depth) + 1
        total_shots += shots_array[j]
        total_calls += D*shots_array[j]

        circuit = circuits[j]
        result = job.result()
        counts_circuit = result.get_counts(circuit)
        zerosd = results_to_good_count_func(counts_circuit)

        zeros.append(zerosd)
    zeros = np.array(zeros)
    if not (kappa_params is None):
        theta_est, kappa_est = max_log_likelihood_kappa(grover_depths, eps, \
            kappa_params, shots_array, zeros)
        probability_est = probability_from_theta(theta_est)
        return probability_est, kappa_est, total_shots, total_calls, shots
    else:
        theta_est = max_log_likelihood(grover_depths, eps, shots_array, zeros)
        probability_est = probability_from_theta(theta_est)
        return probability_est, total_shots, total_calls, shots


def log_likelihood_kappa(grover_depths, eps, kappa_params, shots, zeros):
    eps_use = eps/3
    ones = shots - zeros
    thetas = np.arange(eps_use, np.pi/2, eps_use)
    kappa_max, kappa_fineness = kappa_params
    kappas = np.arange(0.0, kappa_max+kappa_fineness, kappa_fineness)
    ll = np.zeros((thetas.shape[0], kappas.shape[0]))
    for j, grover_depth in enumerate(grover_depths):
        D = (2*grover_depth) + 1
        zterm = 0.5 - (0.5*np.exp(-kappas*grover_depth)*\
            np.cos(2*D*thetas[:, None]))
        oterm = 1.0 - zterm
        ll += zeros[j]*np.log(zterm)
        ll += ones[j]*np.log(oterm)
    return thetas, kappas, ll

def max_log_likelihood_kappa(grover_depths, eps, kappa_params, shots, zeros):
    thetas, kappas, ll = log_likelihood_kappa(grover_depths, eps, \
        kappa_params, shots, zeros)
    idx = np.unravel_index(np.argmax(ll), ll.shape)
    theta = thetas[idx[0]]
    kappa = kappas[idx[1]]
    return theta, kappa