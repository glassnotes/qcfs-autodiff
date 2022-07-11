"""
Simple implementation of a variational eigensolver.
"""

import pennylane as qml
from pennylane import numpy as np


def vqe(qnode, init_params, true_energy, tol=1e-8):
    """Run the VQE for a given ansatz and initial parameters.

    Args:
        qnode (qml.QNode): A QNode that measures an expectation value.
        init_params (array[float]): The starting variational parameters.
        true_energy (float): The expected energy, found analytically. This
            is used to ensure convergence to the proper parameter
        tol (float): The desired accuracy of the results.

    Returns:
        array[float]: The optimal variational parameters that yield the
        ground state of the ansatz.
    """
    N_iter = 1000  # Max iterations

    opt = qml.GradientDescentOptimizer(stepsize=0.1)

    params = init_params.copy()

    for step in range(N_iter):
        params = opt.step(qnode, params)
        new_energy = qnode(params)

        if np.abs(new_energy - true_energy) < tol:
            break

        if step % 50 == 0:
            print(f"At optimization step {step}")

    return params, new_energy
