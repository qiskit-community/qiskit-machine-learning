# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Test Sampling QNN."""
import itertools

from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import ddt, idata, unpack
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler
from qiskit.utils import algorithm_globals

from qiskit_machine_learning.neural_networks import SamplingQNN

INTERPRET_TYPES = ["interpret_1d", "interpret_2d", "none"]
SAMPLERS = ["default", "none"]
SHOTS = [1, 5]
SAMPLES = [1, 2]


@ddt
class TestSamplingQNN(QiskitMachineLearningTestCase):
    """Sampling QNN Tests."""

    def setUp(self) -> None:
        super().setUp()
        algorithm_globals.random_seed = 123456

        def interpret_1d(x):
            return sum((s == "1" for s in f"{x:0b}")) % 2

        def interpret_2d(x):
            return np.asarray([interpret_1d(x), 2 * interpret_1d(x)])

        self.properties = {
            "default": Sampler(),
            "none": None,
            "interpret_1d": interpret_1d,
            "interpret_2d": interpret_2d,
        }

    @idata(itertools.product(SAMPLERS, INTERPRET_TYPES, SAMPLES, SHOTS))
    @unpack
    def test_sampling_qnn_shapes(self, sampler_type, interpret_type, num_samples, num_shots):
        """Test correctness of output shapes of the QNN."""
        qc = QuantumCircuit(2)
        input_parameter = Parameter("x")
        qc.rx(input_parameter, 0)
        qc.cx(0, 1)
        weight_parameter = Parameter("w")
        qc.ry(weight_parameter, 1)

        qnn = SamplingQNN(
            circuit=qc,
            sampler=self.properties[sampler_type],
            num_shots=num_shots,
            input_params=[input_parameter],
            weight_params=[weight_parameter],
            interpret=self.properties[interpret_type],
        )
        input_data = np.ones((num_samples, 1))
        output = qnn.forward(input_data=input_data, weights=1)
        self.assertIsNotNone(output)
        self.assertIsInstance(output, np.ndarray)
        if interpret_type in ("interpret_1d", "none"):
            interpret_shape = (1,)
        elif interpret_type == "interpret_2d":
            interpret_shape = (2,)
        else:
            raise ValueError(f"Unsupported interpret: {interpret_type}")

        output_shape = (num_shots, *interpret_shape)
        self.assertEqual(qnn.output_shape, output_shape)
        self.assertEqual(output.shape, (num_samples, *output_shape))

        output = qnn.backward(input_data=input_data, weights=1)
        self.assertEqual(output, (None, None))

    def test_sampling_qnn_distribution(self):
        """Test correctness of the sample distribution produced by the QNN."""
        qc = QuantumCircuit(2)

        qc.ry(Parameter("0"), 0)
        qc.ry(Parameter("1"), 1)
        qc.rx(np.pi / 2, 0)

        qnn = SamplingQNN(circuit=qc, num_shots=100, input_params=[], weight_params=qc.parameters)
        output = qnn.forward(input_data=[], weights=[0, 0])
        num_zeros = np.sum(output == 0)
        num_ones = np.sum(output == 1)
        num_twos = np.sum(output == 2)
        num_threes = np.sum(output == 3)

        self.assertGreater(num_ones, 30)
        self.assertGreater(num_zeros, 30)
        self.assertEqual(num_twos, 0)
        self.assertEqual(num_threes, 0)
        self.assertEqual(num_ones + num_zeros, 100)

    def test_defaults(self):
        """test the QNN with the default (or minimal) configuration."""
        qc = QuantumCircuit(1)
        qnn = SamplingQNN(circuit=qc)

        output = qnn.forward(input_data=[], weights=[])
        print(output)

    def test_exceptions(self):
        """Test the QNN raises exceptions."""
        qc = QuantumCircuit(1)
        with self.assertRaises(ValueError):
            _ = SamplingQNN(circuit=qc, num_shots=0)
