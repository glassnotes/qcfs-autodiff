"""
Circuits and Hamiltonians for the 4-spin system.
"""

import pennylane as qml
from pennylane import numpy as np


def reduced_H_0():
    """Non-r-dependent part of reduced Hamiltonian."""
    obs = [
        qml.PauliX(0),
        qml.PauliX(1),
        qml.PauliX(0) @ qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliX(1),
        qml.PauliX(0) @ qml.PauliX(1),
        qml.PauliY(0) @ qml.PauliY(1),
    ]
    coeffs = [-1, -1, -1, +1, -np.sqrt(2), -np.sqrt(2)]

    return qml.Hamiltonian(coeffs, obs)


def reduced_H_1(r):
    """r-dependent part of reduced Hamiltonian."""
    obs = [qml.PauliZ(0), qml.PauliZ(1)]
    coeffs = [-2 * r, -2 * r]
    return qml.Hamiltonian(coeffs, obs)


def reduced_H(r):
    """Full reduced Hamiltonian."""
    return reduced_H_0() + reduced_H_1(r)


def variational_ansatz(params):
    """Variational ansatz needed to produce ground states for the 4-spin case."""
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[1, 0])
    qml.RY(params[2], wires=0)
