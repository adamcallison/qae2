from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import MCMT
import numpy as np

def build_ccx_from_cx():
    """Generate ccx gate from cnot and single-qubit gates
    """
    qc_qubits = QuantumRegister(3, 'q')
    qc = QuantumCircuit(qc_qubits)

    qc.h(0)
    qc.cx(1, 0)
    qc.tdg(0)
    qc.cx(2, 0)
    qc.t(0)
    qc.cx(1, 0)
    qc.tdg(0)
    qc.cx(2, 0)
    qc.t(0)
    qc.h(0)
    qc.t(1)
    qc.cx(2, 1)
    qc.tdg(1)
    qc.cx(2, 1)
    qc.t(2)
    return qc

def build_cccx_from_toffolis(ugate_only=False, barriers=False):
    """Generate cccx gate from toffoli, cnot and single-qubit gates

    Parameters
    ----------
    ugate_only : bool, optional
        If True, only the generic 'u' gate is used for single-qubit gates
    barriers : bool, optional
        If True, barriers will be added to circuit for clarity
    """
    qc_qubits = QuantumRegister(4, 'q')
    qc = QuantumCircuit(qc_qubits)

    if ugate_only:
        qc.u(np.pi/2, 0.0, np.pi, 0)
    else:
        qc.h(0)

    qc.ccx(2, 1, 0)
    if ugate_only:
        qc.u(0.0, 0.0, -np.pi/4, 0)
    else:
        qc.tdg(0)
    qc.ccx(3, 1, 0)
    if ugate_only:
        qc.u(0.0, 0.0, np.pi/4, 0)
    else:
        qc.t(0)

    qc.ccx(2, 1, 0)
    if ugate_only:
        qc.u(0.0, 0.0, -np.pi/4, 0)
    else:
        qc.tdg(0)
    qc.ccx(3, 1, 0)
    if ugate_only:
        qc.u(0.0, 0.0, np.pi/4, 0)
    else:
        qc.t(0)

    if ugate_only:
        qc.u(np.pi/2, 0.0, np.pi, 0)
    else:
        qc.h(0)

    if barriers: qc.barrier()

    if ugate_only:
        qc.u(0.0, 0.0, np.pi/4, 1)
    else:
        qc.t(1)
    qc.ccx(3, 2, 1)
    if ugate_only:
        qc.u(0.0, 0.0, -np.pi/4, 1)
    else:
        qc.tdg(1)
    qc.ccx(3, 2, 1)

    if barriers: qc.barrier()

    qc.u(0.0, 0.0, np.pi/8, 2)
    qc.cx(3, 2)
    qc.u(0.0, 0.0, -np.pi/8, 2)
    qc.cx(3, 2)

    if barriers: qc.barrier()

    qc.u(0.0, 0.0, np.pi/8, 3)

    return qc

def build_mcnot(n, base='toffoli', ugate_only=False):
    """Generate a multi-controlled not gate

    Parameters
    ----------
    n : int
        The number of controls. If n=1, cnot is used. If n=2, toffoli is used.
        If n=3 and base='toffoli', a toffoli decomposition of cccx is used.
        In all other cases, MCMT is used
    base : str, optional, 'toffoli' or 'cccx'
    ugate_only : bool, optional
        If True, only the generic 'u' gate is used for single-qubit gates
    """
    base = base.lower()
    qc = QuantumCircuit(n+1)
    if n == 1:
        qc.cx(1, 0)
    elif n == 2:
        qc.ccx(2, 1, 0)
    elif n == 3 and base == 'toffoli':
        qc = qc.compose(build_cccx_from_toffolis(ugate_only=ugate_only), \
            list(range(n+1)))
    else:
        qc = qc.compose(MCMT('x', n, 1), list(range(n+1))[::-1])
    return qc

def build_mcnot_decomp(n, base='cccx_toffoli', ugate_only=False):
    """Generate a multi-controlled not gate via a recursive decomposition with
    ancillas

    Parameters
    ----------
    n : int
        The number of controls.
    base : str, optional, 'cccx_toffoli', 'toffoli' or 'cccx'. 
        The base of the recursive strategy.
        If base='cccx_toffoli', cccx is used as the base, but it is then 
        decomposed directly to toffolis.
    ugate_only : bool, optional
        If True, only the generic 'u' gate is used for single-qubit gates
    """
    base = base.lower()
    if base in ('cccx_toffoli', 'toffoli_cccx'):
        recursive_base, decomp_base = 'cccx', 'toffoli'
    elif base == 'cccx':
        recursive_base, decomp_base = 'cccx', 'cccx'
    elif base == 'toffoli':
        recursive_base, decomp_base = 'toffoli', 'toffoli'
    else:
        raise ValueError

    if ((recursive_base == 'cccx') and (n <= 3)) or \
        ((recursive_base == 'toffoli') and (n <= 2)):
        return build_mcnot(n, base=decomp_base, ugate_only=ugate_only)

    if recursive_base == 'cccx':
        topsize, bottomsize = 3, n-2
    if recursive_base == 'toffoli':
        topsize, bottomsize = 2, n-1
    qc_top = build_mcnot_decomp(topsize, base=base, ugate_only=ugate_only)
    qc_bottom = build_mcnot_decomp(bottomsize, base=base, ugate_only=ugate_only)

    total_qubits = len(qc_bottom.qubits) + topsize
    total_ancilla_qubits = len(qc_bottom.ancillas) + 1
    total_data_qubits = total_qubits - total_ancilla_qubits
    qc_qubits = QuantumRegister(total_data_qubits)
    if total_ancilla_qubits > 0:
        qc_ancilla = AncillaRegister(total_ancilla_qubits)
        qc = QuantumCircuit(qc_qubits, qc_ancilla)
    else:
        qc = QuantumCircuit(qc_qubits)

    if recursive_base == 'cccx':
        composequbits_top = [total_qubits-1, total_data_qubits-3, \
            total_data_qubits-2, total_data_qubits-1]
        composequbits_bottom = list(range(0, total_data_qubits-3))+\
        (list(range(total_data_qubits, total_qubits))[::-1])
    if recursive_base == 'toffoli':
        composequbits_top = [total_qubits-1,total_data_qubits-2,\
        total_data_qubits-1]
        composequbits_bottom = list(range(0, total_data_qubits-2))+\
        (list(range(total_data_qubits, total_qubits))[::-1])
    qc = qc.compose(qc_top, qubits=composequbits_top)
    qc = qc.compose(qc_bottom, qubits=composequbits_bottom)
    qc = qc.compose(qc_top, qubits=composequbits_top)
    return qc

def append_mcnot(qc, target, controls, base='cccx_toffoli', ugate_only=False, \
    reusable_ancillas=None, new_ancilla_name=None):
    """Append a recursively-decomposed multi-controlled not gate to a circuit

    Parameters
    ----------
    qc : quantum circuit
        The circuit to append the mcnot to
    target : int
        The index of the target qubit
    controls : iterable of int
        The indices of the control qubits
    base : str, optional, 'cccx_toffoli', 'toffoli' or 'cccx'. 
        The base of the recursive strategy.
    ugate_only : bool, optional
        If True, only the generic 'u' gate is used for single-qubit gates
    reusable_ancillas : list of int, optional
        Indices of existing ancilla qubits that may be reused
    new_ancilla_name : str, optional
        Name of new ancilla register if one must be added

    Returns
    -------
    qc : quantum circuit
        Copy of input qc with the mcnot appended
    """
    qc = qc.copy()
    if reusable_ancillas is None:
        reusable_ancillas = []
    nancillas_existing = len(reusable_ancillas)
    n = len(controls)
    mcx = build_mcnot_decomp(n, base=base, ugate_only=ugate_only)
    nancillas = len(mcx.ancillas)
    nancillas_extra = np.max((0, nancillas - nancillas_existing))
    if nancillas_extra > 0:
        if new_ancilla_name is None:
            qc_ancillas_extra = AncillaRegister(nancillas_extra)
        else:
            qc_ancillas_extra = AncillaRegister(nancillas_extra, \
                new_ancilla_name)
        qc.add_register(qc_ancillas_extra)
        current_nqubits = len(qc.qubits)
        composequbits = [target] + list(controls) + list(reusable_ancillas) + \
            list(range(current_nqubits-nancillas_extra, current_nqubits))
    else:
        composequbits = [target] + list(controls) + \
            list(reusable_ancillas)[:nancillas]
    qc = qc.compose(mcx, qubits=composequbits)
    return qc