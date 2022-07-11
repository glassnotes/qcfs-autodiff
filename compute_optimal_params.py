"""
This script runs VQE to compute the optimal variational parameters
for the 4- and 6-spin problem. It outputs two files,
 - optimal_params_4.npy
 - optimal_params_6.npy
"""

from functools import partial

import pennylane as qml
from pennylane import numpy as np
from src.vqe import vqe


def diagonalize_hamiltonian(H, n_qubits):
    """Diagonalizes an n_qubits Hamiltonian to determine the
    ground state energy.

    Args:
        H (qml.Hamiltonian): A Hamiltonian.
        n_qubits (int): The number of qubits in the Hamiltonian H.

    Returns:
        float: The lowest-energy eigenvalue found through exact diagonalization
        of the matrix representation of H.
    """
    mat = np.zeros((2 ** n_qubits, 2 ** n_qubits), dtype=np.complex128)

    for coeff, pauli in zip(H.coeffs, H.ops):
        mat += coeff * qml.grouping.pauli_word_to_matrix(
            pauli, wire_map={x: x for x in range(n_qubits)}
        )

    return np.linalg.eigvalsh(mat)[0]


r_values = np.arange(0.5, 1.5, 0.1)

for L in [4, 6]:
    if L == 4:
        n_qubits = 2
        n_params = 3
        from src.spins_4 import reduced_H, variational_ansatz
    elif L == 6:
        n_qubits = 3
        n_params = 7
        from src.spins_6 import reduced_H, variational_ansatz

    # Fully analytical device
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def ground_state_qnode(obs, params):
        variational_ansatz(params)
        return qml.expval(obs)

    diagonalized_ground_state_energies = []
    ground_state_energies = []
    ground_state_params = []

    print(f"Computing optimal parameters for L = {L}")

    for idx, r in enumerate(r_values):
        print(f"\nr = {r:.1f}")

        obs = reduced_H(r)

        # Get the true ground state energy for this Hamiltonian
        true_ground_state_energy = diagonalize_hamiltonian(obs, n_qubits)

        # Run the VQE
        partial_qnode = partial(ground_state_qnode, obs)

        # We initialize all parameters to 0
        init_params = np.array([0.0] * n_params, requires_grad=True)

        optimal_params, ground_state_energy = vqe(
            partial_qnode, init_params, true_ground_state_energy
        )

        # Save our results
        diagonalized_ground_state_energies.append(true_ground_state_energy)
        ground_state_params.append(optimal_params)
        ground_state_energies.append(ground_state_energy)

    print("r\tTrue\t\tObtained")
    for idx, r in enumerate(r_values):
        print(
            f"{r:.1f}\t{diagonalized_ground_state_energies[idx]:.8f}\t{ground_state_energies[idx]:.8f}"
        )
    print("\n\n")

    np.save(f"optimal_params_{L}.npy", np.array(ground_state_params))
