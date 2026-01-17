# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Regression Kernel Loss Functions"""

import unittest
from functools import partial
from test import QiskitMachineLearningTestCase
from ddt import ddt, data
import numpy as np
from scipy.optimize import minimize

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.utils.loss_functions import SVRLoss, MSRLoss, MARLoss, HuberLoss
from qiskit_machine_learning.algorithms.regressors import QSVR


@ddt
class TestRegressionKernelLoss(QiskitMachineLearningTestCase):
    """Test Regression Kernel Loss Functions Algorithm"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 10598
        data_block = zz_feature_map(2)
        trainable_block = zz_feature_map(2, parameter_prefix="θ")
        self.training_parameters = trainable_block.parameters

        self.feature_map = data_block.compose(trainable_block).compose(data_block)

        self.sample_train = np.asarray(
            [
                [3.07876080, 1.75929189],
                [6.03185789, 5.27787566],
                [6.22035345, 2.70176968],
                [0.18849556, 2.82743339],
            ]
        )
        # Simple linear targets for testing
        self.label_train = np.sum(self.sample_train, axis=1)

        self.sample_test = np.asarray([[2.199114860, 5.15221195], [0.50265482, 0.06283185]])
        self.label_test = np.sum(self.sample_test, axis=1)

    @data(SVRLoss, MSRLoss, MARLoss, HuberLoss)
    def test_regression_loss_fit(self, loss_type):
        """Test trainer with regression loss functions."""
        quantum_kernel = TrainableFidelityQuantumKernel(
            feature_map=self.feature_map,
            training_parameters=self.training_parameters,
        )
        loss = loss_type()
        optimizer = partial(minimize, method="COBYLA", options={"maxiter": 10})
        qkt = QuantumKernelTrainer(quantum_kernel=quantum_kernel, loss=loss, optimizer=optimizer)
        qkt_result = qkt.fit(self.sample_train, self.label_train)

        # Ensure user parameters are bound to real values
        self.assertTrue(np.all(qkt_result.quantum_kernel.parameter_values))

        # Ensure it works with QSVR
        qsvr = QSVR(quantum_kernel=qkt_result.quantum_kernel)
        qsvr.fit(self.sample_train, self.label_train)
        score = qsvr.score(self.sample_test, self.label_test)

        # We don't necessarily expect a high score with only 10 iterations,
        # but we expect it to be a finite number.
        self.assertTrue(np.isfinite(score))


if __name__ == "__main__":
    unittest.main()
