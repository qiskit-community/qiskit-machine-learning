# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test QuantumKernelTrainer """
from __future__ import annotations

import unittest
from functools import partial

from test import QiskitMachineLearningTestCase

from ddt import ddt, data
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import ZZFeatureMap

from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_machine_learning.kernels import (
    TrainableFidelityQuantumKernel,
    TrainableFidelityStatevectorKernel,
)
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.utils.loss_functions import SVCLoss


@ddt
class TestQuantumKernelTrainer(QiskitMachineLearningTestCase):
    """Test QuantumKernelTrainer Algorithm

    Tests usage with ``TrainableFidelityQuantumKernel`` and ``TrainableFidelityStatevectorKernel``.
    """

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 10598
        data_block = ZZFeatureMap(2)
        trainable_block = ZZFeatureMap(2, parameter_prefix="θ")
        training_parameters = trainable_block.parameters

        self.feature_map = data_block.compose(trainable_block).compose(data_block)
        self.training_parameters = training_parameters

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
        self.label_test = np.asarray([1, 0])

    @data(
        TrainableFidelityQuantumKernel,
        TrainableFidelityStatevectorKernel,
    )
    def test_default_fit(self, trainable_kernel_type):
        """Test trainer with default parameters."""
        quantum_kernel = trainable_kernel_type(
            feature_map=self.feature_map,
            training_parameters=self.training_parameters,
        )
        qkt = QuantumKernelTrainer(quantum_kernel=quantum_kernel)
        qkt_result = qkt.fit(self.sample_train, self.label_train)

        self._fit_and_assert_score(qkt_result)

    @data(
        TrainableFidelityQuantumKernel,
        TrainableFidelityStatevectorKernel,
    )
    def test_fit_with_params(self, trainable_kernel_type):
        """Test trainer with custom parameters."""
        quantum_kernel = trainable_kernel_type(
            feature_map=self.feature_map,
            training_parameters=self.training_parameters,
        )
        loss = SVCLoss(C=0.8, gamma="auto")
        optimizer = partial(minimize, method="COBYLA", options={"maxiter": 25})
        qkt = QuantumKernelTrainer(quantum_kernel=quantum_kernel, loss=loss, optimizer=optimizer)
        qkt_result = qkt.fit(self.sample_train, self.label_train)

        # Ensure user parameters are bound to real values
        self.assertTrue(np.all(qkt_result.quantum_kernel.parameter_values))

        self._fit_and_assert_score(qkt_result)

    @data(
        TrainableFidelityQuantumKernel,
        TrainableFidelityStatevectorKernel,
    )
    def test_fit_with_no_params(self, trainable_kernel_type):
        """Test trainer with custom parameters."""
        quantum_kernel = trainable_kernel_type(
            feature_map=self.feature_map,
            training_parameters=None,
        )
        loss = SVCLoss(C=0.8, gamma="auto")
        optimizer = partial(minimize, method="COBYLA", options={"maxiter": 25})
        qkt = QuantumKernelTrainer(quantum_kernel=quantum_kernel, loss=loss, optimizer=optimizer)
        with self.assertRaises(ValueError):
            qkt.fit(self.sample_train, self.label_train)

    @data(
        TrainableFidelityQuantumKernel,
        TrainableFidelityStatevectorKernel,
    )
    def test_asymmetric_trainable_parameters(self, trainable_kernel_type):
        """Test when the number of trainable parameters does not equal to the number of features."""
        qc = QuantumCircuit(2)
        training_parameters = Parameter("θ")
        qc.ry(training_parameters, [0, 1])
        feature_params = ParameterVector("x", 2)
        qc.rz(feature_params[0], 0)
        qc.rz(feature_params[1], 1)

        quantum_kernel = trainable_kernel_type(
            feature_map=qc,
            training_parameters=[training_parameters],
        )

        qkt = QuantumKernelTrainer(quantum_kernel=quantum_kernel)
        qkt_result = qkt.fit(self.sample_train, self.label_train)

        self._fit_and_assert_score(qkt_result)

    def _fit_and_assert_score(self, qkt_result):
        # Ensure kernel training functions and is deterministic
        qsvc = QSVC(quantum_kernel=qkt_result.quantum_kernel)
        qsvc.fit(self.sample_train, self.label_train)
        score = qsvc.score(self.sample_test, self.label_test)
        self.assertGreaterEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
