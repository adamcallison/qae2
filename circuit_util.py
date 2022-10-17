from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, \
    ClassicalRegister, transpile
import mcnot

iqs_bg = ['u1', 'u2', 'u3', 'u', 'p', 'r', 'rx', 'ry', 'rz', 'cx', 'cy', 'cz', \
    'csx', 'cp', 'id', 'x', 'y', 'z', 'h', 's', 'sdg', 'sx', 't', 'tdg', \
    'swap', 'ccx', 'cswap', 'unitary', 'diagonal', 'initialize', 'cu1', 'cu2', \
    'cu3', 'rxx', 'ryy', 'rzz', 'rzx', 'mcx', 'mcy', 'mcz', 'mcsx', 'mcp', \
    'mcu1', 'mcu2', 'mcu3', 'mcrx', 'mcry', 'mcrz', 'mcr', 'mcswap', \
    'multiplexer', 'kraus', 'delay', 'roerror']

def zero_projector_circuit(n):
    """Generates the circuit for the zero projection operator

    Parameters
    ----------
    n : int
        number of qubits
    """
    qc_qubits = QuantumRegister(n)
    qc = QuantumCircuit(qc_qubits)
    for j in range(n):
        qc.x(j)
    qc.h(0)
    qc = mcnot.append_mcnot(qc, 0, list(range(1, n)), \
        new_ancilla_name='ancilla', base='toffoli')
    qc.h(0)
    for j in range(n):
        qc.x(j)
    return qc

def Q_circuit(A, O):
    """Generates Q operator circuit for QAE

    Parameters
    ----------
    A : quantum circuit
        The algorithm whose amplitude is being estimated
    O : quantum circuit
        The good-state oracle for the qae algorithm

    Returns
    -------
    qc : quantum circuit
        The generated Q operator circuit
    """
    # not currently supporting ancillae on A or O
    nqubits = len(A.qubits)
    Ai = A.inverse()

    zpc = zero_projector_circuit(nqubits)
    nancillas = len(zpc.ancillas)

    qc_qubits = QuantumRegister(nqubits)
    if nancillas > 0:
        qc_ancillas = AncillaRegister(nancillas)
        qc = QuantumCircuit(qc_qubits, qc_ancillas)
    else:
        qc = QuantumCircuit(qc_qubits)

    qc = qc.compose(O, qubits=list(range(nqubits)))
    qc = qc.compose(Ai, qubits=list(range(nqubits)))
    qc = qc.compose(zpc)
    qc = qc.compose(A, qubits=list(range(nqubits)))
    return qc

def qae_circuits(A, O, grover_depths, measure_qubits, compile_to=None):
    """Generates MLQAE circuits

    Parameters
    ----------
    A : quantum circuit
        The algorithm whose amplitude is being estimated
    O : quantum circuit
        The good-state oracle for the qae algorithm
    grover_depths: iterable
        The grover depth (number of Q applications) to generate circiuits for
    measure_qubits: iterable of int
        The qubit indices to measure at the end of circuits
    compile_to: backend, optinal
        The backend to compile to. If None, just comes to 'u' and 'cx' gates

    Returns
    -------
    circuits : list of quantum circuit
        The generated circuits in the same order as grover_depths
    """
    # not currently supporting ancillae on A or O
    nqubits = len(A.qubits)
    Qqc = Q_circuit(A, O)
    nancillas = len(Qqc.ancillas)
    nclbits = len(measure_qubits)

    qc_qubits = QuantumRegister(nqubits, 'qubits')
    qc_classical = ClassicalRegister(nclbits, 'clbits')
    if nancillas > 0:
        qc_ancillas = AncillaRegister(nancillas, 'ancillas')
        qc = QuantumCircuit(qc_qubits, qc_ancillas, qc_classical)
    else:
        qc = QuantumCircuit(qc_qubits, qc_classical)

    qc = qc.compose(A, qubits=list(range(nqubits)))

    circuits = []
    # currently not the most efficient
    for grover_depth in grover_depths:
        qcgd = qc.copy(name=f'groverdepth{grover_depth}')
        for j in range(grover_depth):
            qcgd = qcgd.compose(Qqc)
        for jid, j in enumerate(measure_qubits):
            qcgd.measure(j, jid)
        if not compile_to is None:
            qcgd = transpile(qcgd, backend=compile_to)
        else:
            qcgd = transpile(qcgd, basis_gates=iqs_bg)
        circuits.append(qcgd)
    return circuits