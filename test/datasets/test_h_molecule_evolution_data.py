# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test H Molecule Evolution Data """

from test import QiskitMachineLearningTestCase

import unittest
import numpy as np
from ddt import ddt, unpack, idata

from qiskit.quantum_info import Statevector
from qiskit.providers.exceptions import QiskitBackendNotFoundError

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.exceptions import IBMInputValueError
from qiskit_ibm_runtime.accounts.exceptions import AccountNotFoundError

from qiskit_machine_learning.datasets import h_molecule_evolution_data


@ddt
class TestHMoleculeEvolution(QiskitMachineLearningTestCase):
    """H Molecule Evolution Tests"""

    @idata([("H2", 4), ("H3", 6)])
    @unpack
    def test_default_params(self, molecule, n_qubits):
        """Checking for right shapes and labels"""
        psi_hf, x_train, y_train, x_test, y_test = h_molecule_evolution_data(
            delta_t=1.0, train_end=2, test_start=4, test_end=6, molecule=molecule
        )

        np.testing.assert_array_equal(psi_hf.shape, (2**n_qubits,))
        np.testing.assert_array_equal(x_train.shape, (3,))
        np.testing.assert_array_equal(x_test.shape, (3,))
        np.testing.assert_array_equal(y_train.shape, (3, 2**n_qubits, 1))
        np.testing.assert_array_equal(y_test.shape, (3, 2**n_qubits, 1))

    @idata([("H2",), ("H3",)])
    @unpack
    def test_statevector_formatting_noiseless(self, molecule):
        """Check if output values are normalized qiskit.circuit_info.Statevector objects"""
        psi_hf, x_tr, y_tr, x_te, y_te = h_molecule_evolution_data(
            1.0,
            1,
            3,
            4,
            molecule=molecule,
            formatting="statevector",
            noise_mode="noiseless",
        )
        self.assertIsInstance(psi_hf, Statevector)
        self.assertTrue(all(isinstance(sv, Statevector) for sv in y_tr))
        self.assertTrue(all(isinstance(sv, Statevector) for sv in y_te))
        self.assertAlmostEqual(psi_hf.probabilities().sum(), 1.0, places=7)
        for sv in y_tr[:2] + y_te[:2]:
            self.assertAlmostEqual(sv.probabilities().sum(), 1.0, places=7)
        np.testing.assert_array_equal(x_tr.shape, (2,))
        np.testing.assert_array_equal(x_te.shape, (2,))
        self.assertEqual(len(y_tr), 2)
        self.assertEqual(len(y_te), 2)

    def test_connecting_to_runtime(self):
        """Fetches the best runtime and connects to its noise model"""
        try:
            service = QiskitRuntimeService()
            backend = service.backends(min_num_qubits=4, operational=True, simulator=False)[0]
        except (IBMInputValueError, QiskitBackendNotFoundError, AccountNotFoundError):
            self.skipTest("IBMQ account or internet unavailable")
        psi_hf, _, y_tr, _, y_te = h_molecule_evolution_data(
            1.0,
            1,
            2,
            3,
            molecule="H2",
            noise_mode=backend.name,
            formatting="ndarray",
        )
        np.testing.assert_array_equal(psi_hf.shape, (2**4,))
        self.assertEqual(y_tr.shape[-1], 1)
        self.assertEqual(y_te.shape[-1], 1)

    def test_error_raises(self):
        """Check if parameter errors are handled"""
        valid = dict(
            delta_t=0.1,
            train_end=5,
            test_start=6,
            test_end=10,
            molecule="H2",
            noise_mode="noiseless",
            formatting="ndarray",
        )

        with self.assertRaises(ValueError):  # bad delta_t
            h_molecule_evolution_data(**{**valid, "delta_t": 0})

        with self.assertRaises(ValueError):  # bad train_end
            h_molecule_evolution_data(**{**valid, "train_end": 0})

        with self.assertRaises(ValueError):  # bad test_start
            h_molecule_evolution_data(**{**valid, "test_start": 0})

        with self.assertRaises(ValueError):  # test_end ≤ test_start
            h_molecule_evolution_data(**{**valid, "test_end": 5})

        with self.assertRaises(ValueError):  # unsupported molecule
            h_molecule_evolution_data(**{**valid, "molecule": "H6"})

        with self.assertRaises(ValueError):  # bad formatting
            h_molecule_evolution_data(**{**valid, "formatting": "json"})

        # invalid backend name – ValueError _or_ RuntimeError depending on connectivity
        with self.assertRaises((ValueError, RuntimeError)):
            h_molecule_evolution_data(**{**valid, "noise_mode": "bad_backend"})


if __name__ == "__main__":
    unittest.main()
