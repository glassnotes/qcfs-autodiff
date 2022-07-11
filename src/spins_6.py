"""
Circuits and Hamiltonians for the 6-spin system.
"""

import pennylane as qml
from pennylane import numpy as np


def reduced_H_0():
    """Non-r-dependent part of reduced Hamiltonian."""
    obs = [
        qml.PauliX(wires=[1]),
        qml.PauliZ(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliZ(wires=[2]),
        qml.PauliX(wires=[1]) @ qml.PauliX(wires=[2]),
        qml.PauliY(wires=[1]) @ qml.PauliY(wires=[2]),
        qml.PauliX(wires=[0]) @ qml.PauliX(wires=[1]),
        qml.PauliY(wires=[0]) @ qml.PauliY(wires=[1]),
        qml.PauliX(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliX(wires=[2]),
        qml.PauliX(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliY(wires=[2]),
    ]

    coeffs = [
        -2.6389584337646843,
        0.1894686909815062,
        -1,
        -1,
        -1 / np.sqrt(2),
        -1 / np.sqrt(2),
        -1,
        -1,
    ]

    return qml.Hamiltonian(coeffs, obs)


def reduced_H_1(r):
    """r-dependent part of reduced Hamiltonian."""
    obs = [
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(2),
        qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]),
    ]
    coeffs = [-3 * r, -r, -r, -r]
    return qml.Hamiltonian(coeffs, obs)


def reduced_H(r):
    """Full reduced Hamiltonian."""
    return reduced_H_0() + reduced_H_1(r)


def variational_ansatz(params):
    """Variational ansatz needed to produce ground states for the 6-spin case."""
    qml.broadcast(qml.RY, wires=range(3), pattern="single", parameters=params[:3])
    qml.broadcast(qml.CNOT, wires=range(3), pattern="ring")
    qml.broadcast(qml.RY, wires=range(3), pattern="single", parameters=params[3:6])
    qml.CNOT(wires=[0, 1])
    qml.RY(params[-1], wires=1)


def variational_ansatz_manila(params):
    """Variational ansatz needed to produce ground states for the 6-spin case,
    manually adjusted to fit the Manila coupling map, which is linear.

    This needed to be incorporated by hand because we need to be able to apply
    error mitigation to the circuit *after* it is adjusted for the coupling map.
    We cannot do this if the coupling map is handled at the device level by
    PennyLane-Qiskit. In principle we could use qml.transforms.transpile, which
    performs simple SWAP-based routing given a coupling map; however this doesn't
    currently support measurement of Hamiltonians or tensor product observables;
    so we are left with doing this manually. Note that:
     - it yield a circuit with the same number of CNOTs as Qiskit's routing for
       the coupling map
     - the SWAP has been placed between qubits 1 and 2 which correspond to qubits
       with the lowest CNOT error rate after mapping; so more CNOTs occur between
       lower-error-rate qubits
    """
    qml.broadcast(qml.RY, wires=range(3), pattern="single", parameters=params[:3])
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 1])
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 1])
    qml.CNOT(wires=[1, 2])
    qml.broadcast(qml.RY, wires=range(3), pattern="single", parameters=params[3:6])
    qml.CNOT(wires=[0, 1])
    qml.RY(params[-1], wires=1)
