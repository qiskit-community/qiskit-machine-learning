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
from ddt import ddt, data

from qiskit import Aer, QuantumCircuit
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.utils import optionals

from qiskit.opflow import PauliSumOp
from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.algorithms.effective_dimension import (
    EffectiveDimension,
    LocalEffectiveDimension,
)


@ddt
class TestEffDim(QiskitMachineLearningTestCase):
    """Test the Effective Dimension algorithm."""

    def setUp(self):
        super().setUp()

        # fix seeds
        algorithm_globals.random_seed = 1234
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

        self.circuit_qnn_1 = CircuitQNN(
            qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            interpret=parity,
            output_shape=2,
            sparse=False,
            quantum_instance=qi_sv,
        )

        # qnn2 for checking result without parity
        self.circuit_qnn_2 = CircuitQNN(
            qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            sparse=False,
            quantum_instance=qi_sv,
        )

        # OpflowQNN
        observable = PauliSumOp.from_list([("Z" * num_qubits, 1)])
        self.opflow_qnn = TwoLayerQNN(
            num_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            observable=observable,
            quantum_instance=qi_sv,
        )

        # define sample numbers
        self.n_list = [5000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000]
        self.n = 5000

    @data(
        # num_inputs, num_params
        ("circuit1", 10, 10, 4.62355184),
        ("circuit1", 1, 1, 1.39529449),
        ("circuit1", 10, 1, 4.92825034),
        ("circuit2", 10, 10, 5.93064171),
    )
    @unpack
    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_alg_results(self, qnn_name, num_inputs, num_params, result):
        """Test that the algorithm results match the original code's."""

        if qnn_name == "circuit2":
            qnn = self.circuit_qnn_2
        else:
            qnn = self.circuit_qnn_1

        global_ed = EffectiveDimension(
            qnn=qnn, num_params=num_params, num_inputs=num_inputs, seed=0
        )

        effdim = global_ed.get_effective_dimension(self.n)

        self.assertAlmostEqual(effdim, result, 5)

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_qnn_type(self):
        """Test that the results are equivalent for opflow and circuit qnn."""

        num_inputs, num_params = 1, 1
        qnn1 = self.circuit_qnn_1
        qnn2 = self.opflow_qnn

        global_ed1 = EffectiveDimension(
            qnn=qnn1,
            num_params=num_params,
            num_inputs=num_inputs,
        )

        global_ed2 = EffectiveDimension(
            qnn=qnn2,
            num_params=num_params,
            num_inputs=num_inputs,
        )

        effdim1 = global_ed1.get_effective_dimension(self.n)
        effdim2 = global_ed2.get_effective_dimension(self.n)

        self.assertAlmostEqual(effdim1, effdim2, 5)

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_custom_data(self):
        """Test that the results are equivalent for equal custom and generated data."""

        num_inputs, num_params = 10, 10
        qnn = self.circuit_qnn_1
        np.random.seed(0)
        inputs = np.random.normal(0, 1, size=(10, qnn.num_inputs))
        np.random.seed(0)  # if seed not set again, test fails
        params = np.random.uniform(0, 1, size=(10, qnn.num_weights))

        global_ed1 = EffectiveDimension(
            qnn=qnn, num_params=num_params, num_inputs=num_inputs, seed=0
        )

        global_ed2 = EffectiveDimension(qnn=qnn, params=params, inputs=inputs, seed=0)

        effdim1 = global_ed1.get_effective_dimension(self.n)
        effdim2 = global_ed2.get_effective_dimension(self.n)

        np.testing.assert_array_equal(global_ed1._inputs, global_ed2._inputs)
        np.testing.assert_array_equal(global_ed1._params, global_ed2._params)

        self.assertAlmostEqual(effdim1, effdim2, 5)

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_multiple_samples(self):
        """Test results for a list of sampling sizes."""

        num_inputs, num_params = 10, 10
        qnn = self.circuit_qnn_1

        global_ed1 = EffectiveDimension(
            qnn=qnn,
            num_params=num_params,
            num_inputs=num_inputs,
        )

        effdim1 = global_ed1.get_effective_dimension(self.n_list)
        effdim2 = global_ed1.get_effective_dimension(np.asarray(self.n_list))

        np.testing.assert_array_equal(effdim1, effdim2)

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_local_ed_error(self):
        """Test that QiskitMachineLearningError is raised for wrong use
        of LocalEffectiveDimension class."""

        with self.assertRaises(ValueError):

            qnn = self.circuit_qnn_1
            inputs = algorithm_globals.random.normal(0, 1, size=(10, qnn.num_inputs))
            params = algorithm_globals.random.uniform(0, 1, size=(10, qnn.num_weights))

            local_ed1 = LocalEffectiveDimension(
                qnn=qnn,
                inputs=inputs,
                params=params,
            )
            local_ed1.get_effective_dimension(self.n)


if __name__ == "__main__":
    unittest.main()
