from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister
from qiskit.compiler import transpile
import mcnot

def zero_projector_circuit(n):
    qc_qubits = QuantumRegister(n, 'q')
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
    # not currently supporting ancillae on A or O
    nqubits = len(A.qubits)
    Ai = A.inverse()

    zpc = zero_projector_circuit(nqubits)
    nancillas = len(zpc.ancillas)

    qc_qubits = QuantumRegister(nqubits, 'q')
    qc_ancillas = AncillaRegister(nancillas, 'ancilla')
    qc = QuantumCircuit(qc_qubits, qc_ancillas)

    qc = qc.compose(O, qubits=list(range(nqubits)))
    qc = qc.compose(Ai, qubits=list(range(nqubits)))
    qc = qc.compose(zpc)
    qc = qc.compose(A, qubits=list(range(nqubits)))
    return qc

def qae_circuits(A, O, grover_depths, measure_qubits, compile_to=None):
    # not currently supporting ancillae on A or O
    nqubits = len(A.qubits)
    Qqc = Q_circuit(A, O)
    nancillas = len(Qqc.ancillas)
    nclbits = len(measure_qubits)

    qc_qubits = QuantumRegister(nqubits, 'q')
    qc_ancillas = AncillaRegister(nancillas, 'ancilla')
    qc_classical = ClassicalRegister(nclbits, 'classical')
    qc = QuantumCircuit(qc_qubits, qc_ancillas, qc_classical)

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
            # need to transpile as AerSimulator doesn't support cry???
            qcgd = transpile(qcgd, basis_gates=['cx', 'u'])
        circuits.append(qcgd)
    return circuits