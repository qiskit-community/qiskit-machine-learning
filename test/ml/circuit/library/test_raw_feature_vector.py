# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the ``RawFeatureVector`` circuit."""

import unittest
from test.ml import QiskitMLTestCase

import numpy as np
from qiskit import transpile, Aer
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.ml.circuit.library import RawFeatureVector

from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.ml.datasets import wine


class TestRawFeatureVector(QiskitMLTestCase):
    """Test the ``RawFeatureVector`` circuit."""

    def test_construction(self):
        """Test creating the circuit works."""

        circuit = RawFeatureVector(4)

        with self.subTest('check number of qubits'):
            self.assertEqual(circuit.num_qubits, 2)

        with self.subTest('check parameters'):
            self.assertEqual(len(circuit.parameters), 4)

        with self.subTest('check unrolling fails'):
            with self.assertRaises(QiskitError):
                _ = transpile(circuit, basis_gates=['u', 'cx'], optimization_level=0)

    def test_fully_bound(self):
        """Test fully binding the circuit works."""

        circuit = RawFeatureVector(8)

        params = np.random.random(8) + 1j * np.random.random(8)
        params /= np.linalg.norm(params)

        bound = circuit.bind_parameters(params)

        ref = QuantumCircuit(3)
        ref.initialize(params, ref.qubits)

        self.assertEqual(bound, ref)

    def test_partially_bound(self):
        """Test partially binding the circuit works."""

        circuit = RawFeatureVector(4)
        params = circuit.ordered_parameters

        with self.subTest('single numeric value'):
            circuit.assign_parameters({params[0]: 0.2}, inplace=True)
            self.assertEqual(len(circuit.parameters), 3)

        with self.subTest('bound to another parameter'):
            circuit.assign_parameters({params[1]: params[2]}, inplace=True)
            self.assertEqual(len(circuit.parameters), 2)

        with self.subTest('test now fully bound circuit'):
            bound = circuit.assign_parameters({params[2]: 0.4, params[3]: 0.8})
            ref = QuantumCircuit(2)
            ref.initialize([0.2, 0.4, 0.4, 0.8], ref.qubits)
            self.assertEqual(bound, ref)

    def test_usage_in_vqc(self):
        """Test using the circuit the a single VQC iteration works."""
        feature_dim = 4
        _, training_input, test_input, _ = wine(training_size=1,
                                                test_size=1,
                                                n=feature_dim,
                                                plot_data=False)
        feature_map = RawFeatureVector(feature_dimension=feature_dim)

        vqc = VQC(COBYLA(maxiter=1),
                  feature_map,
                  EfficientSU2(feature_map.num_qubits, reps=1),
                  training_input,
                  test_input)
        backend = Aer.get_backend('qasm_simulator')
        result = vqc.run(backend)
        self.assertTrue(result['eval_count'] > 0)


if __name__ == '__main__':
    unittest.main()
