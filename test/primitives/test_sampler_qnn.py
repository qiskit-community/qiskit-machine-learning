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

"""Test Sampler QNN with Terra primitives."""
import numpy as np
from test import QiskitMachineLearningTestCase

from qiskit.primitives import Sampler
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit import Aer
from qiskit.utils import QuantumInstance

from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.primitives.sampler_qnn import SamplerQNN

algorithm_globals.random_seed = 42


class TestSamplerQNN(QiskitMachineLearningTestCase):
    """Sampler QNN Tests."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 12345

        # define test circuit
        num_qubits = 3
        self.qc = RealAmplitudes(num_qubits, entanglement="linear", reps=1)
        self.qi_qasm = QuantumInstance(Aer.get_backend("aer_simulator"), shots=10)

    def test_forward_pass(self):

        parity = lambda x: "{:b}".format(x).count("1") % 2
        output_shape = 2  # this is required in case of a callable with dense output

        circuit_qnn = CircuitQNN(
            self.qc,
            input_params=self.qc.parameters[:3],
            weight_params=self.qc.parameters[3:],
            sparse=False,
            interpret=parity,
            output_shape=output_shape,
            quantum_instance=self.qi_qasm,
        )

        inputs = np.asarray(algorithm_globals.random.random(size=(1, circuit_qnn._num_inputs)))
        weights = algorithm_globals.random.random(circuit_qnn._num_weights)
        circuit_qnn_fwd = circuit_qnn.forward(inputs, weights)

        sampler_factory = Sampler
        with SamplerQNN(
            circuit=self.qc,
            input_params=self.qc.parameters[:3],
            weight_params=self.qc.parameters[3:],
            sampler_factory=sampler_factory,
            interpret=parity,
            output_shape=output_shape,
        ) as qnn:

            sampler_qnn_fwd = qnn.forward(inputs, weights)

            np.testing.assert_array_almost_equal(
                np.asarray(sampler_qnn_fwd), np.asarray(circuit_qnn_fwd), 0.1
            )
