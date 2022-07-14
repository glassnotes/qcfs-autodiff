"""
Basic code for differentiable zero-noise extrapolation.

This code is based on the examples provided in https://arxiv.org/abs/2202.13414,
"Quantum computing with differentiable quantum transforms".

Autodiff-friendly implementations are achieved using PennyLane's
math library, which performs framework-agnostic manipulation of tensors
(i.e., the implementation should work independently of whether one is
using PennyLane with JAX, NumPy, Tensorflow, or Torch).

This file contains:
 - Method(s) for Richardson extrapolation (compute_gamma_j, richardson_extrapolation)
 - A method for linear extrapolation (linear_extrapolation)
 - A PennyLane batch transform that applies zero-noise extrapolation using
   a specified noise-application transform and extrapolation method.
"""

from functools import partial
from scipy.optimize import curve_fit
import pennylane as qml
import pennylane.math as math
from pennylane import numpy as np


def compute_gamma_j(scale_factors):
    """Evaluates the Lagrange basis polynomials at 0.

    This function computes the \gamma_j defined in Eq. (25) of the manuscript.

    Args:
        scale_factors (array[float]): The noise scale factors.

    Returns:
        array[float]: Evaluation of basis polynomials to be used for
        Richardson extrapolation.
    """
    gamma_j = math.zeros(len(scale_factors))

    for j, _ in enumerate(scale_factors):
        denoms = scale_factors - scale_factors[j]
        denoms_slice = math.concatenate((denoms[0:j], denoms[j + 1 :]))
        scale_factors_slice = math.concatenate((scale_factors[0:j], scale_factors[j + 1 :]))

        gamma_j[j] = math.prod(scale_factors_slice / denoms_slice)

    return gamma_j


def richardson_extrapolation(scale_factors, noisy_results):
    """Extrapolate to the zero-noise limit using Richardson extrapolation.

    This function implements Eq. (25) of my notes. It has been
    implemented to consider extrapolation of two different types of values:
     1. expectation values
     2. measurement outcome probabilities
    Differences arise due to the dimensions of the incoming results; there is
    probably a nice way to consolidate them using some form of broadcasting of
    the gamma_j values, but the conditional statement works well enough for now.

    Args:
        scale_factors (array[float]): The noise scale factors.
        noisy_results (array[float]): The set of results obtained
            when running the noisy circuits at all the different scale factors.

    Returns:
        float: The estimated value of the fit function at the zero-noise level.
    """
    scale_factors = math.stack(scale_factors)
    gamma_j = compute_gamma_j(scale_factors)

    # Case for expectation values
    if len(math.shape(noisy_results)) == 2:
        noisy_results = math.stack(noisy_results).ravel()
        return math.sum(gamma_j * noisy_results)

    # Case when extrapolating probabilities; there is an extra dimension involved
    noisy_results = [math.stack(res).ravel() for res in noisy_results]
    return [math.sum(gamma_j * res) for res in math.transpose(noisy_results)]


def linear_extrapolation(scale_factors, energies):
    """Extrapolate to the zero-noise limit using a simple linear fit.

    Args:
        scale_factors (array[float]): The noise scale factors.
        energies (array[float]): The set of energy expectation values obtained
            when running the noisy circuits at all the different scale factors.

    Returns:
        float: The estimated value of the linear fit at the zero-noise level.
    """

    scale_factors = math.stack([float(x) for x in scale_factors])
    unwrapped_energies = math.stack(energies).reshape(len(energies))

    # Manually do a simple linear regression in a framework-agnostic manner.
    # https://www.mathsisfun.com/data/least-squares-regression.html
    N = len(energies)

    sum_scales = math.sum(scale_factors)
    sum_energies = math.sum(unwrapped_energies)

    numerator = N * math.sum(scale_factors * unwrapped_energies) - sum_scales * sum_energies
    denominator = N * math.sum(scale_factors**2) - sum_scales**2
    slope = numerator / denominator

    return (sum_energies - slope * sum_scales) / N


@qml.batch_transform
def zne(tape, extrapolation_fn, folding_fn, scale_factors):
    """PennyLane batch transform for ZNE.

    Args:
        extrapolation_fn (Callable): The function that will be used to perform
            the numerical extrapolation. The signature must have the form
            extrapolation_fn(scale_factories, energies), where energies is an
            array of floats of the same size as the scale factors.
        folding_fn (qml.qfunc_transform): A quantum transform that can be used
            to add a scalable amount of noise to the circuit. The signature of
            the transform function must be folding_fn(tape, scale_factor).
        scale_factors (array[float]): The noise scale factors.

    Returns:
        float: The estimated value of the linear fit at the zero-noise level.
    """
    with qml.tape.stop_recording():
        tapes = [folding_fn.tape_fn(tape, scale) for scale in scale_factors]

    processing_fn = partial(extrapolation_fn, scale_factors)

    return tapes, processing_fn
