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

"""Test for Quantum Kernel Training program."""

from test import QiskitMachineLearningTestCase

import numpy as np

from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.circuit.library import ZZFeatureMap
from qiskit.algorithms.optimizers import COBYLA

from qiskit_machine_learning.runtime import QuantumKernelTrainerClient
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.utils.loss_functions import SVCLoss
from .fake_qkt_runtime import FakeQKTRuntimeProvider


def generate_feature_map():
    """
    Create a 2 qubit circuit consisting of 2 user parameters and 2 data bound parameters.
    """
    data_block = ZZFeatureMap(2)
    trainable_block = ZZFeatureMap(2)
    user_parameters = trainable_block.parameters

    for i, _ in enumerate(user_parameters):
        user_parameters[i]._name = f"θ[{i}]"

    feature_map = data_block.compose(trainable_block).compose(data_block)

    return feature_map, user_parameters


class TestQKTRuntimeClient(QiskitMachineLearningTestCase):
    """Test the quantum-kernel-trainer program."""

    def setUp(self):
        super().setUp()
        self._provider = FakeQKTRuntimeProvider()
        self.shots = 10
        self.sample_train = np.asarray(
            [
                [3.07876080, 1.75929189],
                [6.03185789, 5.27787566],
                [6.22035345, 2.70176968],
                [0.18849556, 2.82743339],
            ]
        )
        self.label_train = np.asarray([0, 0, 1, 1])

        self.sample_test = np.asarray([[2.199114860, 5.15221195], [0.50265482, 0.06283185]])
        self.label_test = np.asarray([0, 1])

        self.feature_map, self.user_parameters = generate_feature_map()

    def test_fit_kernel(self):
        """Test kernel training"""
        backend = QasmSimulatorPy()
        quantum_kernel = QuantumKernel(
            feature_map=self.feature_map,
            user_parameters=self.user_parameters,
            quantum_instance=backend,
        )
        optimizer = COBYLA(maxiter=25)
        loss_func = SVCLoss().get_variational_callable(
            quantum_kernel=quantum_kernel, data=self.sample_train, labels=self.label_train
        )
        provider = self._provider

        qkt_program = QuantumKernelTrainerClient(
            quantum_kernel=quantum_kernel,
            backend=backend,
            loss=loss_func,
            optimizer=optimizer,
            shots=self.shots,
            provider=provider,
        )
        qkt_program.fit_kernel(self.sample_train, self.label_train)
