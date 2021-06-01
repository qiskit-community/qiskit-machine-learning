# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
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

from test import QiskitMachineLearningTestCase

import numpy as np
from qiskit import transpile, Aer
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Statevector
from qiskit.utils import QuantumInstance, algorithm_globals

from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.circuit.library import RawFeatureVector


class TestRawFeatureVector(QiskitMachineLearningTestCase):
    """Test the ``RawFeatureVector`` circuit."""

    def test_construction(self):
        """Test creating the circuit works."""

        circuit = RawFeatureVector(4)

        with self.subTest("check number of qubits"):
            self.assertEqual(circuit.num_qubits, 2)

        with self.subTest("check parameters"):
            self.assertEqual(len(circuit.parameters), 4)

        with self.subTest("check unrolling fails"):
            with self.assertRaises(QiskitError):
                _ = transpile(circuit, basis_gates=["u", "cx"], optimization_level=0)

    def test_fully_bound(self):
        """Test fully binding the circuit works."""

        circuit = RawFeatureVector(8)

        params = np.random.random(8) + 1j * np.random.random(8)
        params /= np.linalg.norm(params)

        bound = circuit.bind_parameters(params)

        ref = QuantumCircuit(3)
        ref.initialize(params, ref.qubits)

        self.assertEqual(bound.decompose(), ref)

    def test_partially_bound(self):
        """Test partially binding the circuit works."""

        circuit = RawFeatureVector(4)
        params = circuit.parameters

        with self.subTest("single numeric value"):
            circuit.assign_parameters({params[0]: 0.2}, inplace=True)
            self.assertEqual(len(circuit.parameters), 3)

        with self.subTest("bound to another parameter"):
            circuit.assign_parameters({params[1]: params[2]}, inplace=True)
            self.assertEqual(len(circuit.parameters), 2)

        with self.subTest("test now fully bound circuit"):
            bound = circuit.assign_parameters({params[2]: 0.4, params[3]: 0.8})
            ref = QuantumCircuit(2)
            ref.initialize([0.2, 0.4, 0.4, 0.8], ref.qubits)
            self.assertEqual(bound.decompose(), ref)

    def test_usage_in_vqc(self):
        """Test using the circuit the a single VQC iteration works."""

        # specify quantum instance and random seed
        algorithm_globals.random_seed = 12345
        quantum_instance = QuantumInstance(
            Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        # construct data
        num_samples = 10
        num_inputs = 4
        X = algorithm_globals.random.random(  # pylint: disable=invalid-name
            (num_samples, num_inputs)
        )
        y = 1.0 * (np.sum(X, axis=1) <= 2)
        while len(np.unique(y, axis=0)) == 1:
            y = 1.0 * (np.sum(X, axis=1) <= 2)
        y = np.array([y, 1 - y]).transpose()

        feature_map = RawFeatureVector(feature_dimension=num_inputs)
        ansatz = RealAmplitudes(feature_map.num_qubits, reps=1)
        # classification may fail sometimes, so let's fix initial point
        initial_point = np.array([0.5] * ansatz.num_parameters)

        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=10),
            quantum_instance=quantum_instance,
            initial_point=initial_point,
        )

        vqc.fit(X, y)
        score = vqc.score(X, y)
        self.assertGreater(score, 0.5)

    def test_bind_after_composition(self):
        """Test binding the parameters after the circuit was composed onto a larger one."""
        circuit = QuantumCircuit(2)
        circuit.h([0, 1])

        raw = RawFeatureVector(4)
        circuit.append(raw, [0, 1])

        bound = circuit.bind_parameters([1, 0, 0, 0])

        self.assertTrue(Statevector.from_label("00").equiv(bound))


if __name__ == "__main__":
    unittest.main()
