# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Opflow QNN."""

import unittest

from test import QiskitMachineLearningTestCase

import numpy as np
from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance

from qiskit_machine_learning.neural_networks import CircuitQNN


class TestCircuitQNN(QiskitMachineLearningTestCase):
    """Opflow QNN Tests."""

    def setUp(self):
        super().setUp()

        # specify "run configuration"
        backend = Aer.get_backend('statevector_simulator')
        quantum_instance = QuantumInstance(backend)

        # define QNN
        num_qubits = 2
        feature_map = ZZFeatureMap(num_qubits)
        var_form = RealAmplitudes(num_qubits, reps=1)

        qc = QuantumCircuit(2)
        qc.append(feature_map, range(2))
        qc.append(var_form, range(2))

        input_params = list(feature_map.parameters)
        weight_params = list(var_form.parameters)

        def parity(x):
            return (-1)**sum(x)

        self.qnn = CircuitQNN(qc, input_params, weight_params,
                              interpret=parity, quantum_instance=quantum_instance)

    def test_circuit_qnn1(self):
        """Opflow QNN Test."""

        input_data = np.zeros(self.qnn.num_inputs)
        weights = np.zeros(self.qnn.num_weights)

        result = self.qnn.probabilities(input_data, weights)
        print(result)


if __name__ == '__main__':
    unittest.main()
