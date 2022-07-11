"""
(To run this file it must be moved to the parent directory.)

Full computation of energies, second energy derivatives, and
fidelity susceptibility.

To run this file, use:
  python mit_experiments_shots.py <L> <n_trials> <folding_fn> <scale_factors>
where
 - <L> is the number of spins
 - <n_trials> is an integer representing the number of trials at each r
 - <folding_fn> is either unitary, cnot, or none
 - <scale_factors> is a string of digits or none, e.g., 135, meaning to perform
   mitigation with scale factors 1, 3, and 5.

The following file containing the optimal variational parameters is required:
 - optimal_params_{L}.npy
"""
import sys
from functools import partial

import pennylane as qml
from pennylane import numpy as np

import concurrent.futures as futures

from src.fidelity_susceptibility import (
    compute_second_energy_derivative,
    compute_fidelity_susceptibility,
)

from qiskit.providers.aer import QasmSimulator
from qiskit.transpiler import CouplingMap

from src.zne import zne, richardson_extrapolation
from src.folding import cnot_folding, unitary_folding

# Set this directory to the one you wish the results to be stored in
dir_prefix = "."


def parse_command_line_arguments(args):
    """Parse command line arguments for simulation and mitigation configuration.

    Args:
        args (List[str]): Should be a 5-element list where 
            - the element at index 1 is the system size
            - index 2 contains the number of trials
            - indices 3 and 4 contain mitigation information. The first should
              be either "unitary" or "cnot", and the second should be a string
              of integers representing the set of scale factors to use.

    Returns:
        int, int, Callable, array[float]: The system size, number of trials,
        mitigation function to use, and the set of numerical scale factors.
    """

    if len(args) != 5:
        print("Please run this program as:")
        print(f"python mit_experimentsspin.py <L> <n_trials> <folding_fn> <scale_factors>")
        sys.exit()

    try:
        L = int(args[1])
        if L not in [4, 6]:
            raise ValueError
    except ValueError:
        print("Input L must be either 4 or 6.")
        sys.exit()
        
    try:
        n_trials = int(args[2])
    except TypeError:
        print("First argument should be an integer number of trials.")
        sys.exit()

    folding_options_dict = {"unitary": unitary_folding, "cnot": cnot_folding, "none": None}
    if args[3] not in list(folding_options_dict.keys()):
        raise ValueError("Folding function must be either 'unitary', 'cnot', or 'none'")
    folding_fn = folding_options_dict[args[3]]

    if folding_fn is not None:
        try:
            scale_factors = np.array([float(x) for x in str(args[4])])
        except ValueError:
            print("Scale factor must be a string of integers. For example, ")
            print("'135' will be converted to scale factors [1.0, 3.0, 5.0].")
            sys.exit()
    else:
        scale_factors = None

    return L, n_trials, folding_fn, scale_factors

if __name__ == "__main__":
    ###########################################################
    # Set up basic simulation parameters based on system size #
    ###########################################################
    L, N_trials, folding_fn, scale_factors = parse_command_line_arguments(sys.argv)
    
    r_values = np.arange(0.5, 1.5, 0.1)
    ground_state_params = np.load(f"optimal_params_{L}.npy")

    if L == 4:
        initial_layout = [3, 4, 0, 1, 2] # Qubits 3/4 have the lowest CNOT error
        n_qubits = 2  # Number of qubits required to represent reduced Ising Hamiltonian
        n_params = 3  # Number of variational parameters
        from src.spins_4 import reduced_H_1, reduced_H, variational_ansatz
    if L == 6:
        initial_layout = [2, 3, 4, 0, 1] # Next lowest CNOT error between qubits 2/3 so use those
        n_qubits = 3
        n_params = 7
        from src.spins_6 import reduced_H_1, reduced_H
        from src.spins_6 import variational_ansatz_manila as variational_ansatz # Optimized for arch


    #####################################################################
    # Set up the processor; not currently configurable via command line #
    #####################################################################
    dev = qml.device(
        "qiskit.aer", backend="qasm_simulator", wires=5, shots=8192,
    )

    coupling_map = CouplingMap([(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)])

    dev.set_transpile_args(
        **{"optimization_level": 0, "initial_layout": initial_layout, "coupling_map": coupling_map}
    )
    
    ##############################################
    # Configuration options for error mitigation #
    ##############################################
    if folding_fn is None and scale_factors is None:
        mitigation_config = None
        folding_fn_name = "unmitigated"
    else:
        mitigation_config = {
            "extrapolation_fn": richardson_extrapolation,
            "folding_fn": folding_fn,
            "scale_factors": scale_factors,
            "dev": dev,
        }
        folding_fn_name = sys.argv[3] + sys.argv[4]


    #################################
    # Compute ground state energies #
    #################################    
    def ground_state_circuit(obs, params):
        """A simple circuit that computes the energy of an observable after
        applying the variational ansatz.
        
        Args:
            obs (qml.Hamiltonian): a Hamiltonian.
            params (array[float]): the variational parameters.
        
        Returns:
            float: the expectation value of obs.
        """
        variational_ansatz(params)
        return qml.expval(obs)

    ground_state_qnode = qml.QNode(ground_state_circuit, dev, diff_method="parameter-shift")
    
    print("Estimating ground state energies")

    def _compute_ground_state_energies(r_info):
        """Function to compute ground state energies for multiple trials for a given
        r. Implemented like this so it can be called by each thread independently."""
        obs = reduced_H(r_info[1])

        _ = ground_state_qnode(obs, ground_state_params[r_info[0]])
        ham_tapes, ham_fn = qml.transforms.hamiltonian_expand(ground_state_qnode.qtape, group=False)

        gs_this_r = []

        if mitigation_config is not None:
            zne_tapes, zne_fn = qml.transforms.map_batch_transform(
                partial(
                    zne,
                    extrapolation_fn=mitigation_config["extrapolation_fn"],
                    folding_fn=mitigation_config["folding_fn"],
                    scale_factors=mitigation_config["scale_factors"],
                ),
                ham_tapes,
            )

            for _ in range(N_trials):
                gs_this_r.append(ham_fn(zne_fn(qml.execute(zne_tapes, dev, gradient_fn=None))))

        else:
            for _ in range(N_trials):
                gs_this_r.append(ham_fn(qml.execute(ham_tapes, dev, gradient_fn=None)))

        return gs_this_r

    # We run in parallel; 10 workers, one per R
    with futures.ProcessPoolExecutor(max_workers=10) as pool:
        ground_state_energies = pool.map(_compute_ground_state_energies, list(enumerate(r_values)))

    ground_state_energies = np.array(list(ground_state_energies))

    
    #####################################
    # Compute second energy derivatives #
    #####################################
    print("Estimating second energy derivatives. This will take some time.")

    def _compute_second_energy_derivative_at_r(r_info):
        """Function to compute second energy derivative for multiple trials for a given r."""        
        this_dtheta_drs = []
        this_d2E_dr2s = []
        this_cond = []

        for _ in range(N_trials):
            d2E_dr2, dtheta_dr, cond = compute_second_energy_derivative(
                ground_state_qnode,
                ground_state_params[r_info[0]],
                reduced_H_1,
                reduced_H,
                r_info[1],
                mitigation_config,
            )

            this_d2E_dr2s.append(d2E_dr2)
            this_dtheta_drs.append(dtheta_dr)
            this_cond.append(cond)

        return this_dtheta_drs, this_d2E_dr2s, this_cond

    with futures.ProcessPoolExecutor(max_workers=10) as pool:
        unprocessed_d2E_drs = pool.map(
            _compute_second_energy_derivative_at_r, list(enumerate(r_values))
        )

    unprocessed_d2E_drs = list(unprocessed_d2E_drs)
    mitigated_dtheta_drs = np.array([x[0] for x in unprocessed_d2E_drs])
    d2E_dr2s = np.array([x[1] for x in unprocessed_d2E_drs])
    conds = np.array([x[2] for x in unprocessed_d2E_drs])
    
    ###################################
    # Compute fidelity susceptibility #
    ###################################
    def overlap_circuit(optimal_params, params):
        """A circuit to estimate the ground state fidelity.

        Args:
            optimal_params (array[float]): A non-differentiable array
                representing the optimal variational parameters to produce the
                ground state.
            params (array[float]): A differentiable set of parameters at which
                to compute the overlap with the ground state.

        Returns:
            array[float]: Measurement outcome probabilities; the outcome probability
            of the all-zeros state is the absolute value squared of the overlap.
        """
        variational_ansatz(optimal_params)
        qml.adjoint(variational_ansatz)(params)
        return qml.probs(wires=dev.wires)

    overlap_qnode = qml.QNode(overlap_circuit, dev, diff_method="parameter-shift")

    print("Estimating fidelity susceptibility")

    def _compute_fidelity_susceptibility_at_r(r_info):
        """Function to compute fidelity susceptibility for multiple trials for a given r."""          
        fidelity_susceptibility_at_r = []

        for idx_trial in range(N_trials):
            fidelity_susceptibility_at_r.append(
                compute_fidelity_susceptibility(
                    overlap_qnode,
                    ground_state_params[r_info[0]],
                    mitigated_dtheta_drs[r_info[0]][idx_trial],
                    mitigation_config,
                )
            )

        return fidelity_susceptibility_at_r

    with futures.ProcessPoolExecutor(max_workers=10) as pool:
        fidelity_susceptibility = pool.map(
            _compute_fidelity_susceptibility_at_r, list(enumerate(r_values))
        )

    fidelity_susceptibility = np.abs(np.array(list(fidelity_susceptibility)))

    
    ###################################
    # Save and output all the results #
    ###################################
    np.save(
        f"{dir_prefix}/results-{L}spin_{folding_fn_name}_second_energy_derivative.npy",
        -d2E_dr2s / L,
    )
    np.save(
        f"{dir_prefix}/results-{L}spin_{folding_fn_name}_conds.npy",
        conds
    )        
    np.save(
        f"{dir_prefix}/results-{L}spin_{folding_fn_name}_dtheta_dr.npy",
        mitigated_dtheta_drs,
    )    
    np.save(
        f"{dir_prefix}/results-{L}spin_{folding_fn_name}_ground_state_energy.npy",
        ground_state_energies,
    )
    np.save(
        f"{dir_prefix}/results-{L}spin_{folding_fn_name}_fidelity_susceptibility.npy",
        fidelity_susceptibility / L,
    )

    # Dump all relevant output to a file as well
    outfile_name = f"{dir_prefix}/stdout-{L}spin_{folding_fn_name}.txt"

    with open(outfile_name, "w") as outfile:
        outfile.write(f"L={L}")
        outfile.write("\n\nFolding function \n")
        outfile.write(str(folding_fn))
        outfile.write("\n\nScale factors \n")
        outfile.write(str(scale_factors))
        outfile.write("\n\nGround state params \n")
        outfile.write(str(ground_state_params))
        outfile.write("\n\n-second energy derivatives / L\n")
        outfile.write(str(-d2E_dr2s / L))
        outfile.write("\n\n-std dev second energy derivatives / L\n")
        outfile.write(str(np.std(-d2E_dr2s / L, axis=1)))
        outfile.write("\n\n-fidelity susceptibility / L\n")
        outfile.write(str(fidelity_susceptibility / L))
        outfile.write("\n\n-std dev fidelity susceptibility / L\n")
        outfile.write(str(np.std(fidelity_susceptibility / L, axis=1)))
