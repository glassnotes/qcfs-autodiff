"""
Methods to apply scaled noise to a quantum circuit for error mitigation.

This code is based on the examples provided in https://arxiv.org/abs/2202.13414,
"Quantum computing with differentiable quantum transforms".

This file contains methods for:
 - CNOT pair insertion (cnot_folding)
 - Unitary folding (unitary_folding)
"""

import pennylane as qml


@qml.qfunc_transform
def cnot_folding(tape, scale_factor):
    """Quantum function transform to perform CNOT pair insertion.

    For each CNOT in the circuit, a number of additional CNOT pairs
    (derived from the scale factor) is added in order to systematically scale
    up the amount of noise present.

    Args:
        qfunc (function): A quantum function
        scale_factor (float): Scale factor of the noise. The number of folds
            is computed as the nearest integer to (scale_factor - 1) / 2.

    Returns:
        function: the transformed quantum function
    """
    num_pairs = qml.math.round((scale_factor - 1.0) / 2.0)

    for op in tape:
        qml.apply(op)

        if op.name == "CNOT":
            for _ in range(int(2 * num_pairs)):
                qml.apply(op)


@qml.qfunc_transform
def unitary_folding(tape, scale_factor):
    """Quantum function transform to perform unitary folding.

    The provided tape is run once in the normal direction, then some number
    of repetitions of its adjoint, followed by the original operation.

    Args:
        qfunc (function): A quantum function
        scale_factor (float): Scale factor of the noise. The number of folds
            is computed as the nearest integer to (scale_factor - 1) / 2.

    Returns:
        function: the transformed quantum function
    """
    # First run is "forwards".
    for op in tape.operations:
        qml.apply(op)

    num_folds = qml.math.round((scale_factor - 1.0) / 2.0)

    # Now we do the folds
    for _ in range(int(num_folds)):
        # Go through in reverse and apply the adjoints
        for op in tape.operations[::-1]:
            op.adjoint()

        # Go through forwards again
        for op in tape.operations:
            qml.apply(op)

    # Apply the measurements normally
    for m in tape.measurements:
        qml.apply(m)
