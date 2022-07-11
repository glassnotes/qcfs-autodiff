from itertools import chain

import pytest

import pennylane as qml
from pennylane import numpy as np

from folding import cnot_folding, unitary_folding
from spins_6 import variational_ansatz, variational_ansatz_manila


def qfunc(theta):
    qml.RX(theta[0], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.CRY(theta[1], wires=[1, 2])
    qml.CNOT(wires=[0, 2])

theta = np.array([1.0, 2.0])
        

class Test6SpinAnsatz:
    @pytest.mark.parametrize(
        "params",
        [
            np.zeros(7),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            np.array([0.2, 0.6, -0.3, 0.2, -0.7, 1.9, 0.33]),
        ],
    )
    def test_ansatz_equivalence(self, params):
        """Tests that the hand-optimized ansatz does the same thing as the original
        for the 6-spin case."""
        original_matrix = qml.matrix(variational_ansatz)(params)
        new_matrix = qml.matrix(variational_ansatz_manila)(params)
        assert np.allclose(original_matrix, new_matrix)

    
class TestCNOTFolding:
    def test_identity_scalefactor(self):
        """Test that if scale factor is 1, CNOT folding leaves a circuit as-is."""
        folded_qfunc = cnot_folding(1.0)(qfunc)

        original_ops = qml.transforms.make_tape(qfunc)(theta).operations
        folded_ops = qml.transforms.make_tape(folded_qfunc)(theta).operations

        assert len(original_ops) == len(folded_ops)
        assert all([o_op.name == f_op.name for o_op, f_op in zip(original_ops, folded_ops)])

    @pytest.mark.parametrize("scale_factor", [3.0, 5.0])
    def test_scalefactor(self, scale_factor):
        """Test that CNOT folding applies the correct number of CNOTs."""
        original_ops = qml.transforms.make_tape(qfunc)(theta).operations
        
        folded_qfunc = cnot_folding(scale_factor)(qfunc)
        folded_ops = qml.transforms.make_tape(folded_qfunc)(theta).operations
    
        assert len(folded_ops) == 2 + 2 + 2 * (int(scale_factor) - 1)

        expected_ops = list(chain.from_iterable([
            [op.name] if op.name != "CNOT" else ["CNOT"] * (int(scale_factor)) for op in original_ops
        ]))

        assert all([f_op.name == e_op for f_op, e_op in zip(folded_ops, expected_ops)])
        

class TestUnitaryFolding:
    def test_identity_scalefactor(self):
        """Test that if scale factor is 1, unitary folding leaves a circuit as-is."""
        folded_qfunc = unitary_folding(1.0)(qfunc)

        original_ops = qml.transforms.make_tape(qfunc)(theta).operations
        folded_ops = qml.transforms.make_tape(folded_qfunc)(theta).operations

        assert len(original_ops) == len(folded_ops)
        assert all([o_op.name == f_op.name for o_op, f_op in zip(original_ops, folded_ops)])

    @pytest.mark.parametrize("scale_factor", [3.0, 5.0])
    def test_scalefactor(self, scale_factor):
        """Test that unitary folding applies the correct number of folds and takes adjoint."""
        original_ops = qml.transforms.make_tape(qfunc)(theta).operations
        original_names = [[op.name] for op in original_ops]
        original_params = [op.data[0] for op in original_ops if op.num_params > 0]
        
        folded_qfunc = unitary_folding(scale_factor)(qfunc)
        folded_ops = qml.transforms.make_tape(folded_qfunc)(theta).operations
        folded_names = [op.name for op in folded_ops]
        folded_params = [op.data[0] for op in folded_ops if op.num_params > 0]
        
        assert len(folded_ops) == scale_factor * len(original_ops)
        assert len(folded_params) == scale_factor * len(original_params)

        expected_names = original_names.copy()
        for _ in range(int((scale_factor - 1)/2)):
            expected_names.extend(original_names[::-1])
            expected_names.extend(original_names)
        expected_names = list(chain.from_iterable(expected_names))

        expected_params = original_params.copy() 
        for _ in range(int((scale_factor - 1)/2)):
            expected_params.extend([-x for x in original_params[::-1]]) # Adjoint
            expected_params.extend([x for x in original_params]) # Original

        assert all([f_op == e_op for f_op, e_op in zip(folded_names, expected_names)])
        assert np.allclose(np.array(folded_params), np.array(expected_params))
