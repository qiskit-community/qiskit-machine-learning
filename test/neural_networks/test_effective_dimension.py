# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
""" Unit Tests for Effective Dimension Algorithm """

import unittest

from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import ddt, data, unpack

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.utils import QuantumInstance, algorithm_globals, optionals

from qiskit.opflow import PauliSumOp
from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.neural_networks import (
    EffectiveDimension,
    LocalEffectiveDimension,
)
from qiskit_machine_learning import QiskitMachineLearningError


@ddt
class TestEffectiveDimension(QiskitMachineLearningTestCase):
    """Test the Effective Dimension algorithm"""

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def setUp(self):
        super().setUp()

        algorithm_globals.random_seed = 1234
        from qiskit_aer import Aer

        qi_sv = QuantumInstance(
            Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        # set up quantum neural networks
        num_qubits = 3
        feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=1)
        ansatz = RealAmplitudes(num_qubits, reps=1)

        # CircuitQNNs
        qc = QuantumCircuit(num_qubits)
        qc.append(feature_map, range(num_qubits))
        qc.append(ansatz, range(num_qubits))

        def parity(x):
            return f"{x:b}".count("1") % 2

        circuit_qnn_1 = CircuitQNN(
            qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            interpret=parity,
            output_shape=2,
            sparse=False,
            quantum_instance=qi_sv,
        )

        # qnn2 for checking result without parity
        circuit_qnn_2 = CircuitQNN(
            qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            sparse=False,
            quantum_instance=qi_sv,
        )

        # OpflowQNN
        observable = PauliSumOp.from_list([("Z" * num_qubits, 1)])
        opflow_qnn = TwoLayerQNN(
            num_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            observable=observable,
            quantum_instance=qi_sv,
        )

        self.qnns = {"circuit1": circuit_qnn_1, "circuit2": circuit_qnn_2, "opflow": opflow_qnn}

        # define sample numbers
        self.n_list = [5000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000]
        self.n = 5000

    @data(
        # qnn_name, num_inputs, num_weights, result
        ("circuit1", 10, 10, 4.51202148),
        ("circuit1", 1, 1, 1.39529449),
        ("circuit1", 10, 1, 3.97371533),
        ("circuit2", 10, 10, 5.90859124),
    )
    @unpack
    def test_alg_results(self, qnn_name, num_inputs, num_params, result):
        """Test that the algorithm results match the original code's."""
        qnn = self.qnns[qnn_name]
        global_ed = EffectiveDimension(qnn=qnn, weight_samples=num_params, input_samples=num_inputs)

        effdim = global_ed.get_effective_dimension(self.n)

        self.assertAlmostEqual(effdim, result, 5)

    def test_qnn_type(self):
        """Test that the results are equivalent for opflow and circuit qnn."""

        num_input_samples, num_weight_samples = 1, 1
        qnn1 = self.qnns["circuit1"]
        qnn2 = self.qnns["opflow"]

        global_ed1 = EffectiveDimension(
            qnn=qnn1,
            weight_samples=num_weight_samples,
            input_samples=num_input_samples,
        )

        global_ed2 = EffectiveDimension(
            qnn=qnn2,
            weight_samples=num_weight_samples,
            input_samples=num_input_samples,
        )

        effdim1 = global_ed1.get_effective_dimension(self.n)
        effdim2 = global_ed2.get_effective_dimension(self.n)

        self.assertAlmostEqual(effdim1, 1.395, 3)
        self.assertAlmostEqual(effdim1, effdim2, 5)

    def test_multiple_data(self):
        """Test results for a list of sampling sizes."""

        num_input_samples, num_weight_samples = 10, 10
        qnn = self.qnns["circuit1"]

        global_ed1 = EffectiveDimension(
            qnn=qnn,
            weight_samples=num_weight_samples,
            input_samples=num_input_samples,
        )

        effdim1 = global_ed1.get_effective_dimension(self.n_list)
        effdim2 = global_ed1.get_effective_dimension(np.asarray(self.n_list))

        np.testing.assert_array_equal(effdim1, effdim2)

    def test_inputs(self):
        """Test results for different input combinations."""

        qnn = self.qnns["circuit1"]

        num_input_samples, num_weight_samples = 10, 10
        inputs = algorithm_globals.random.uniform(0, 1, size=(num_input_samples, qnn.num_inputs))
        weights = algorithm_globals.random.uniform(0, 1, size=(num_weight_samples, qnn.num_weights))

        global_ed1 = EffectiveDimension(
            qnn=qnn,
            weight_samples=num_weight_samples,
            input_samples=num_input_samples,
        )

        global_ed2 = EffectiveDimension(
            qnn=qnn,
            weight_samples=weights,
            input_samples=inputs,
        )

        effdim1 = global_ed1.get_effective_dimension(self.n_list)
        effdim2 = global_ed2.get_effective_dimension(self.n_list)

        np.testing.assert_array_almost_equal(effdim1, effdim2, 0.2)

    def test_inputs_shapes(self):
        """Test results for different input combinations."""

        qnn = self.qnns["circuit1"]

        num_inputs, num_params = 10, 10
        inputs_ok = algorithm_globals.random.uniform(0, 1, size=(num_inputs, qnn.num_inputs))
        weights_ok = algorithm_globals.random.uniform(0, 1, size=(num_params, qnn.num_weights))

        inputs_wrong = algorithm_globals.random.uniform(0, 1, size=(num_inputs, 1))
        weights_wrong = algorithm_globals.random.uniform(0, 1, size=(num_params, 1))

        with self.assertRaises(QiskitMachineLearningError):
            EffectiveDimension(
                qnn=qnn,
                weight_samples=weights_ok,
                input_samples=inputs_wrong,
            )

        with self.assertRaises(QiskitMachineLearningError):
            EffectiveDimension(
                qnn=qnn,
                weight_samples=weights_wrong,
                input_samples=inputs_ok,
            )

    def test_local_ed_params(self):
        """Test that QiskitMachineLearningError is raised for wrong parameters sizes."""

        qnn = self.qnns["circuit1"]

        num_inputs, num_params = 10, 10
        inputs_ok = algorithm_globals.random.uniform(0, 1, size=(num_inputs, qnn.num_inputs))
        weights_ok = algorithm_globals.random.uniform(0, 1, size=(1, qnn.num_weights))
        weights_ok2 = algorithm_globals.random.uniform(0, 1, size=(qnn.num_weights))
        weights_wrong = algorithm_globals.random.uniform(0, 1, size=(num_params, qnn.num_weights))

        LocalEffectiveDimension(
            qnn=qnn,
            weight_samples=weights_ok,
            input_samples=inputs_ok,
        )

        LocalEffectiveDimension(
            qnn=qnn,
            weight_samples=weights_ok2,
            input_samples=inputs_ok,
        )

        with self.assertRaises(QiskitMachineLearningError):
            LocalEffectiveDimension(
                qnn=qnn,
                weight_samples=weights_wrong,
                input_samples=inputs_ok,
            )


if __name__ == "__main__":
    unittest.main()
