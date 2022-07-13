"""
This script uses the optimal parameter values obtained in the
`compute_optimal_params.py` script to compute and output the
exact values of ground state energy, second energy derivative,
and fidelity susceptibility that we will compare against.

It requires the input files:
 - optimal_params_4.npy
 - optimal_params_6.npy
"""

import pennylane as qml
from pennylane import numpy as np

from src.fidelity_susceptibility import (
    compute_second_energy_derivative,
    compute_fidelity_susceptibility,
)


def diagonalize_hamiltonian(H, n_qubits):
    mat = np.zeros((2**n_qubits, 2**n_qubits), dtype=np.complex128)

    for coeff, pauli in zip(H.coeffs, H.ops):
        mat += coeff * qml.grouping.pauli_word_to_matrix(
            pauli, wire_map={x: x for x in range(n_qubits)}
        )

    return np.linalg.eigvalsh(mat)[0]


r_values = np.arange(0.5, 1.5, 0.1)


for L in [4, 6]:
    if L == 4:
        n_qubits = 2
        from src.spins_4 import reduced_H, reduced_H_1, variational_ansatz
    elif L == 6:
        n_qubits = 3
        from src.spins_6 import reduced_H, reduced_H_1, variational_ansatz

    ground_state_params = np.load(f"optimal_params_{L}.npy")

    # Fully analytical device
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def ground_state_qnode(obs, params):
        variational_ansatz(params)
        return qml.expval(obs)

    @qml.qnode(dev)
    def overlap_qnode(optimal_params, params):
        variational_ansatz(optimal_params)
        qml.adjoint(variational_ansatz)(params)
        return qml.probs(wires=dev.wires)

    # Compute exact values of all the results
    ground_state_energies = []
    second_energy_derivatives = []
    fidelity_susceptibility = []

    for idx, r in enumerate(r_values):
        obs = reduced_H(r)

        # Get the true ground state energy for this Hamiltonian
        ground_state_energy = diagonalize_hamiltonian(obs, n_qubits)
        ground_state_energies.append(ground_state_energy)

        # Compute the second energy derivative
        d2E_dr2, dtheta_dr, _ = compute_second_energy_derivative(
            ground_state_qnode, ground_state_params[idx], reduced_H_1, reduced_H, r
        )
        second_energy_derivatives.append(d2E_dr2)

        # Compute the fidelity susceptibility
        fid_susc = compute_fidelity_susceptibility(
            overlap_qnode, ground_state_params[idx], dtheta_dr
        )
        fidelity_susceptibility.append(fid_susc)

    # Process the results
    second_energy_derivatives = -np.array(second_energy_derivatives) / L
    fidelity_susceptibility = np.abs(np.array(fidelity_susceptibility)) / L

    print(f"\nL={L}")
    print("Ground state energies")
    print(ground_state_energies)
    print("\nSecond energy derivatives")
    print(second_energy_derivatives)
    print("\nFidelity susceptibility")
    print(fidelity_susceptibility)

    np.save(f"exact_gse_L{L}.npy", np.array(ground_state_energies))
    np.save(f"exact_d2e_L{L}.npy", np.array(second_energy_derivatives))
    np.save(f"exact_fs_L{L}.npy", np.array(fidelity_susceptibility))
