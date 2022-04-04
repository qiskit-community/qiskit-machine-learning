# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Effective Dimension Algorithm """

import unittest

from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import ddt, data
from qiskit import Aer, QuantumCircuit
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.utils import QuantumInstance, algorithm_globals

from qiskit.opflow import PauliSumOp
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.algorithms.effective_dimension import (
    EffectiveDimension,
    LocalEffectiveDimension,
)


@ddt
class TestEffDim(QiskitMachineLearningTestCase):
    """Test the Effective Dimension algorithm"""

    def setUp(self):
        super().setUp()

        # fix seeds
        algorithm_globals.random_seed = 12345
        qi_sv = QuantumInstance(
            Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        # set up qnns
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

        # qnn2 for checking result without parity with
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

        # define results
        self.result1 = 4.62355185  # 3,10,10
        self.result2 = 1.39529449  # 3,1,1
        self.result3 = 4.92825035  # 3,10,1
        self.result4 = 5.93064172  # circuitqnn2 with 3,10,10

    @data(
        # num_inputs, num_params
        ("circuit1", 10, 10),
        ("circuit1", 1, 1),
        ("circuit1", 10, 1),
        ("circuit2", 10, 10),
    )
    def test_alg_results(self, config):
        """Test that the algorithm results match the original code's."""

        qnn_name, num_inputs, num_params = config

        if qnn_name == "circuit2":
            qnn = self.circuit_qnn_2
            result = self.result4
        else:
            qnn = self.circuit_qnn_1
            if num_inputs == 1:
                result = self.result2
            elif num_params == 10:
                result = self.result1
            else:
                result = self.result3

        global_ed = EffectiveDimension(
            qnn=qnn, num_params=num_params, num_inputs=num_inputs, fix_seed=True
        )

        effdim = global_ed.get_eff_dim(self.n)

        self.assertAlmostEqual(effdim, result, 5)

    def test_qnn_type(self):
        """Test that the results are equivalent for opflow and circuit qnn."""

        num_inputs, num_params = 1, 1
        qnn1 = self.circuit_qnn_1
        qnn2 = self.opflow_qnn

        global_ed1 = EffectiveDimension(
            qnn=qnn1, num_params=num_params, num_inputs=num_inputs, fix_seed=True
        )

        global_ed2 = EffectiveDimension(
            qnn=qnn2, num_params=num_params, num_inputs=num_inputs, fix_seed=True
        )

        effdim1 = global_ed1.get_eff_dim(self.n)
        effdim2 = global_ed2.get_eff_dim(self.n)

        self.assertAlmostEqual(effdim1, effdim2, 5)

    def test_custom_data(self):
        """Test that the results are equivalent for equal custom and generated data."""
        num_inputs, num_params = 10, 10
        qnn = self.circuit_qnn_1

        np.random.seed(0)
        inputs = np.random.normal(0, 1, size=(num_inputs, qnn.num_inputs))
        np.random.seed(0)
        params = np.random.uniform(0, 1, size=(num_params, qnn.num_weights))

        global_ed1 = EffectiveDimension(
            qnn=qnn, num_params=num_params, num_inputs=num_inputs, fix_seed=True
        )

        global_ed2 = EffectiveDimension(qnn=qnn, params=params, inputs=inputs, fix_seed=True)

        effdim1 = global_ed1.get_eff_dim(self.n)
        effdim2 = global_ed2.get_eff_dim(self.n)

        self.assertTrue(global_ed1._inputs.all() == global_ed2._inputs.all())
        self.assertTrue(global_ed1._params.all() == global_ed2._params.all())
        self.assertAlmostEqual(effdim1, effdim2, 5)

    def test_multiple_samples(self):
        """Test results for a list of sampling sizes."""

        num_inputs, num_params = 10, 10
        qnn = self.circuit_qnn_1

        global_ed1 = EffectiveDimension(
            qnn=qnn, num_params=num_params, num_inputs=num_inputs, fix_seed=True
        )

        effdim1 = global_ed1.get_eff_dim(self.n_list)
        effdim2 = global_ed1.get_eff_dim(np.asarray(self.n_list))

        self.assertTrue(effdim1.all() == effdim2.all())

    def test_local_ed_error(self):
        """Test that QiskitMachineLearningError is raised for wrong use
        of LocalEffectiveDimension class."""

        with self.assertRaises(QiskitMachineLearningError):

            qnn = self.circuit_qnn_1
            np.random.seed(0)
            inputs = np.random.normal(0, 1, size=(10, qnn.num_inputs))
            np.random.seed(0)
            params = np.random.uniform(0, 1, size=(10, qnn.num_weights))

            local_ed1 = LocalEffectiveDimension(
                qnn=qnn, inputs=inputs, params=params, fix_seed=True
            )
            local_ed1.get_eff_dim(self.n)


if __name__ == "__main__":
    unittest.main()
