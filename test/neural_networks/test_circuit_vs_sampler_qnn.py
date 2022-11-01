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

"""Test Sampler QNN vs Circuit QNN."""

from test import QiskitMachineLearningTestCase

import itertools
import unittest
import numpy as np
from ddt import ddt, idata

from qiskit import BasicAer
from qiskit.algorithms.gradients import ParamShiftSamplerGradient
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.opflow import Gradient
from qiskit.primitives import Sampler
from qiskit.utils import QuantumInstance, algorithm_globals

from qiskit_machine_learning.neural_networks import CircuitQNN, SamplerQNN
import qiskit_machine_learning.optionals as _optionals

SPARSE = [True, False]
INPUT_GRADS = [True, False]


@ddt
class TestCircuitQNNvsSamplerQNN(QiskitMachineLearningTestCase):
    """Circuit vs Sampler QNN Tests. To be removed once CircuitQNN is deprecated"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 10598

        self.parity = lambda x: f"{x:b}".count("1") % 2
        self.output_shape = 2  # this is required in case of a callable with dense output

        # define feature map and ansatz
        num_qubits = 2
        feature_map = ZZFeatureMap(num_qubits, reps=1)
        var_form = RealAmplitudes(num_qubits, reps=1)
        # construct circuit
        self.qc = QuantumCircuit(num_qubits)
        self.qc.append(feature_map, range(2))
        self.qc.append(var_form, range(2))

        # store params
        self.input_params = list(feature_map.parameters)
        self.weight_params = list(var_form.parameters)

        self.sampler = Sampler()

    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    @idata(itertools.product(SPARSE, INPUT_GRADS))
    def test_new_vs_old(self, config):
        """Circuit vs Sampler QNN Test. To be removed once CircuitQNN is deprecated"""

        sparse, input_grads = config
        qi_sv = QuantumInstance(BasicAer.get_backend("statevector_simulator"))

        circuit_qnn = CircuitQNN(
            self.qc,
            input_params=self.qc.parameters[:3],
            weight_params=self.qc.parameters[3:],
            sparse=sparse,
            interpret=self.parity,
            output_shape=self.output_shape,
            quantum_instance=qi_sv,
            gradient=Gradient("param_shift"),
            input_gradients=input_grads,
        )

        sampler_qnn = SamplerQNN(
            sampler=self.sampler,
            circuit=self.qc,
            input_params=self.qc.parameters[:3],
            weight_params=self.qc.parameters[3:],
            interpret=self.parity,
            output_shape=self.output_shape,
            gradient=ParamShiftSamplerGradient(self.sampler),
            input_gradients=input_grads,
        )

        inputs = np.asarray(algorithm_globals.random.random(size=(3, circuit_qnn._num_inputs)))
        weights = algorithm_globals.random.random(circuit_qnn._num_weights)

        circuit_qnn_fwd = circuit_qnn.forward(inputs, weights)
        sampler_qnn_fwd = sampler_qnn.forward(inputs, weights)

        diff_fwd = circuit_qnn_fwd - sampler_qnn_fwd
        self.assertAlmostEqual(np.max(np.abs(diff_fwd)), 0.0, places=3)

        circuit_qnn_input_grads, circuit_qnn_weight_grads = circuit_qnn.backward(inputs, weights)
        sampler_qnn_input_grads, sampler_qnn_weight_grads = sampler_qnn.backward(inputs, weights)

        diff_weight = circuit_qnn_weight_grads - sampler_qnn_weight_grads
        self.assertAlmostEqual(np.max(np.abs(diff_weight)), 0.0, places=3)

        if input_grads:
            diff_input = circuit_qnn_input_grads - sampler_qnn_input_grads
            self.assertAlmostEqual(np.max(np.abs(diff_input)), 0.0, places=3)
