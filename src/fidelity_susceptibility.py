"""
Methods for computing the second energy derivative, fidelity susceptibility,
and applying error mitigation to the values involved in computing the above.

As detailed in the manuscript, three types of gradients are required to compute
the second energy derivative:
 1. The gradients of the energy of the parameter (r)-dependent part of the Hamiltonian  
    with respect to the variational parameters (compute_dH1_dtheta).
 2. The Hessian of the full Hamiltonian expectation value with respect to the 
    variational parameters. 
 3. The gradients of the variational parameters with respect to the parameter r
    (compute_theta_dr). To compute these, we require the values obtained in steps 1
    and 2; they are used to set up a linear system which is solved to get dtheta_dr.
The results of dH1_dtheta and dtheta_dr are then combined using the chain rule
to obtain the full second energy derivative (compute_second_energy_derivative).

To compute the fidelity susceptibility, we require:
 1. The values obtained in step 3 of the previous list (dtheta_dr); if we compute the 
    energy derivative first, we should already have this values available, which saves us 
    the trouble of solving for them again. 
 2. The Hessian of the overlap w.r.t. the variational parameters (the overlap circuit
    is constructed using two copies of the variational parameters: one set is trainable,
    and the other is not). This is d2H_dtheta_2.
The results of d2H_dtheta_2 and dtheta/dr are then combined using the chain rule
to obtain the full second energy derivative (compute_fidelity_susceptibility).
"""
from functools import partial

import pennylane as qml
from pennylane import numpy as np

from .zne import zne


def compute_mitigated_gradient(tape, mitigation_config, order=1):
    """Chain batch transforms for gradients and error mitigation to compute
    a mitigated derivative of specified order for either a Hamiltonian expectation
    value, or measurement outcome probability.

    Batch transforms are PennyLane transforms that take a quantum tape as input,
    and return a set of transformed tapes, and a processing function that can be
    used to combine the execution results. See the docs for more details:
    https://pennylane.readthedocs.io/en/stable/code/api/pennylane.batch_transform.html

    Expansion of a Hamiltonian into constituent terms, gradient computation,
    and error mitigation are all individually implemented as batch transforms.
    This function combines them in an order conducive for obtaining error-mitigated
    estimates gradients and Hessians.

    Args:
        tape (qml.QuantumTape): The quantum tape whose parameters we would
            like to compute the gradients of.
        mitigation_config (Dict): Config options for error mitigation. Must
            contain. The following fields must be provided:
             - "extrapolation_fn" (Callable)
             - "folding_fn" (Callable)
             - "scale_factor" (array[float])
             - "dev" (qml.Device)
        order (int): Specifies which order of gradient to take. Default of
            1 corresponds to the parameter-shift gradient; a value of
            2 corresponds to the parameter-shift Hessian.

    Return:
        array[float]: The error-mitigated values of the specified gradient.
    """
    gradient_fn = qml.gradients.param_shift_hessian if order == 2 else qml.gradients.param_shift

    # The way we chain batch transforms and compute the mitigated gradients depends on the
    # type of measurement we are making, and therefore the kind of value we are
    # computing the derivative of. For a Hamiltonian expval, we first expand the initial
    # tape into one tape per expectation value. Then we proceed with the gradient
    # computation and error mitigation.
    meas_type = tape.measurements[0].return_type.value

    if meas_type == "expval":
        ham_tapes, ham_fn = qml.transforms.hamiltonian_expand(tape, group=False)
        grad_tapes, grad_fn = qml.transforms.map_batch_transform(gradient_fn, ham_tapes)
    else:
        grad_tapes, grad_fn = gradient_fn(tape)

    # Maps each gradient tape into a set of tapes (and processing function) that
    # will compute the error-mitigated quantities. The reconstruction function zne_fn will stitch
    # the executed results back together regardless of the measurement type; the expval version
    # will just need some extra post-processing since there is another layer of transforms.
    zne_tapes, zne_fn = qml.transforms.map_batch_transform(
        partial(
            zne,
            extrapolation_fn=mitigation_config["extrapolation_fn"],
            folding_fn=mitigation_config["folding_fn"],
            scale_factors=mitigation_config["scale_factors"],
        ),
        grad_tapes,
    )

    mitigated_grad = grad_fn(
        zne_fn(qml.execute(zne_tapes, mitigation_config["dev"], gradient_fn=None))
    )

    if meas_type == "expval":
        return ham_fn(mitigated_grad)

    return mitigated_grad


def compute_dH1_dtheta(qnode, H1, params, mitigation_config=None):
    """Computes the derivatives of the r-dependent part of the
    Hamiltonian w.r.t. variational parameters.

    Args:
        qnode (qml.QNode): A QNode that computes the energy after
            running the variational circuit preparing the ground state.
            The signature of the QNode should be qnode(obs, params) where
            obs is an observable to be measured.
        H1 (qml.Hamiltonian): The r-dependent part of the Hamiltonian.
        params (array[float]): Values of the variational parameters.
        mitigation_config (None or Dict): If not None, contains the config options
            for error mitigation. See docstring of compute_mitigated_gradient
            for details about what this should contain.

    Returns:
        array[float]: The derivative dH1/dtheta at the specified parameter values.
    """
    # If no mitigation options are configured, we simply compute the gradient
    # using built-in PennyLane functionality
    if mitigation_config is None:
        return qml.grad(qnode)(H1(1.0), params)

    # If mitigation is specified, we evaluate first the QNode to construct the tape,
    # which then gets passed to the mitigation function.
    _ = qnode(H1(1.0), params)
    return compute_mitigated_gradient(qnode.qtape, mitigation_config, order=1)


def compute_dtheta_dr(qnode, H, r, dH1_dtheta, params, mitigation_config=None):
    """Computes the derivatives of the variational parameters with
    respect to the Hamiltonian parameters by solving a linear system.

    Args:
        qnode (qml.QNode): A QNode that computes the energy after
            running the variational circuit preparing the ground state.
            The signature of the QNode should be qnode(obs, params) where
            obs is an observable to be measured.
        H (qml.Hamiltonian): The full problem Hamiltonian.
        r (float): The value of the Hamiltonian parameter.
        dH1_dtheta (array[float]): Pre-computed values of the derivative of
            the r-dependent part of the energy.
        params (array[float]): Values of the variational parameters.
        mitigation_config (None or Dict): If not None, contains the config options
            for error mitigation. See docstring of compute_mitigated_gradient
            for details about what this should contain.

    Returns:
        array[float]: The derivatives dtheta/dr at the specified parameter
        values.
    """
    # First we compute the Hessian of H w.r.t. variational parameters
    # If no mitigation specified, this is easy and can be done using the
    # param_shift_hessian transform directly. Otherwise, we construct the
    # tape and pass it to our mitigated gradient function.
    if mitigation_config is None:
        hess_fn = qml.gradients.param_shift_hessian(qnode)
        d2H_dthetaij = hess_fn(H(r), params)
    else:
        _ = qnode(H(r), params)
        d2H_dthetaij = compute_mitigated_gradient(qnode.tape, mitigation_config, order=2)

    # The derivatives dtheta/dr are obtained by solving the linear system of
    # equations detailed in equation XX of the paper
    return np.linalg.solve(d2H_dthetaij, -dH1_dtheta), np.linalg.cond(d2H_dthetaij)


def compute_second_energy_derivative(qnode, optimal_params, H1, H, r, mitigation_config=None):
    """Compute the second derivative of the energy with respect to the
    variational parameters using the chain rule.

    Args:
        qnode (qml.QNode): A QNode that computes the ground state.
        optimal_params (array[float]): The parameters at the variational
            minimum (ground state).
        H1 (qml.Hamiltonian): r-dependent part of the Hamiltonian.
        H (qml.Hamiltonian): The full Hamiltonian.
        r (float): The Hamiltonian parameter.
        mitigation_config (None or Dict): If not None, contains the config options
            for error mitigation. See docstring of compute_mitigated_gradient
            for details about what this should contain.

    Returns:
        float, array[float]: The second energy derivative, as well as the
        derivatives of the variational parameters with respect to r. We return
        the second set of values because they are used to compute fidelity susceptibility,
        and we only want to evaluate them once, especially if mitigation is required.
    """
    # The two sets of terms from the chain rule: d2E_dr2 = sum_i dH1/dtheta_i dtheta_i/dr
    dH1_dtheta = compute_dH1_dtheta(qnode, H1, optimal_params, mitigation_config)
    dtheta_dr, cond = compute_dtheta_dr(qnode, H, r, dH1_dtheta, optimal_params, mitigation_config)

    d2E_dr2 = np.dot(dH1_dtheta, dtheta_dr)

    return d2E_dr2, dtheta_dr, cond


def compute_fidelity_susceptibility(overlap_qnode, params, dtheta_drs, mitigation_config=None):
    """Compute the fidelity susceptibility through the Hessian of an overlap circuit.

    Args:
        overlap_qnode (qml.QNode): A QNode for an overlap circuit. The circuit
            should have the following signature: overlap_qnode(optimal_params, params)
            where optimal_params are the trainable parameters we are computing the Hessian
            with respect to.
        params (array[float]): Values of the variational parameters.
        dtheta_drs (array[float]): The gradients of the parameters w.r.t. the
            Hamiltonian parameter r. These can be computed separately using the
            compute_dtheta_dr function.
        mitigation_config (None or Dict): If not None, contains the config options
            for error mitigation. See docstring of compute_mitigated_gradient
            for details about what this should contain.

    Returns:
        float: The fidelity susceptibility.
    """
    non_trainable_params = np.array(params, requires_grad=False)

    # Compute the Hessian w.r.t. the output. If no mitigation config, we call
    # the param-shift Hessian function directly and look at the first value
    # (corresponds to gradient of the measurement outcome probability of the |00...0>
    # bitstring, which is what's needed to evalute overlap). Otherwise, we must
    # use our mitigation function.
    if mitigation_config is None:
        hessian = qml.gradients.param_shift_hessian(overlap_qnode)(params, non_trainable_params)[0]
    else:
        _ = overlap_qnode(params, non_trainable_params)
        hessian = compute_mitigated_gradient(overlap_qnode.qtape, mitigation_config, order=2)[0]

    # Evaluate the fidelity susceptibility. This is computed using the chain rule
    # obtained by differentiating the expression for the overlap, with an additional
    # simplification as described in equation XX of the notes.
    fid_susc = 0
    for i in range(hessian.shape[0]):
        for j in range(hessian.shape[1]):
            fid_susc += 0.5 * hessian[i, j] * dtheta_drs[i] * dtheta_drs[j]

    return fid_susc
