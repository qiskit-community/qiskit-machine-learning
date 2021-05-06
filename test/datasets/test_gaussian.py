# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Gaussian """

import unittest
from test import QiskitMachineLearningTestCase
import numpy as np
from qiskit_machine_learning.datasets import gaussian


class TestGaussian(QiskitMachineLearningTestCase):
    """Gaussian tests."""

    def test_gaussian(self):
        """Gaussian test."""

        training_features, training_labels, test_features, test_labels = gaussian(
            training_size=20, test_size=10, n=2, plot_data=False
        )
        np.testing.assert_array_equal(training_features.shape, (40, 2))
        np.testing.assert_array_equal(training_labels.shape, (40, 2))

        np.testing.assert_array_equal(np.sum(training_labels, axis=0), np.array([20, 20]))
        np.testing.assert_array_equal(np.sum(training_labels, axis=1), np.ones(40))

        np.testing.assert_array_equal(test_features.shape, (20, 2))
        np.testing.assert_array_equal(test_features.shape, (20, 2))

        np.testing.assert_array_equal(np.sum(test_labels, axis=0), np.array([10, 10]))
        np.testing.assert_array_equal(np.sum(test_labels, axis=1), np.ones(20))


if __name__ == "__main__":
    unittest.main()
