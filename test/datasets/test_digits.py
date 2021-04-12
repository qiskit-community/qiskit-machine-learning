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

""" Test Digits"""

import unittest
from test import QiskitMachineLearningTestCase
import json
import numpy as np
from qiskit_machine_learning.datasets import digits


class TestDigits(QiskitMachineLearningTestCase):
    """Digits tests."""

    def test_digits(self):
        """Digits test."""

        input_file = self.get_resource_path('training_input.digits',
                                            'datasets')
        with open(input_file) as file:
            training_input_ref = json.load(file)

        input_file = self.get_resource_path('test_input.digits',
                                            'datasets')
        with open(input_file) as file:
            test_input_ref = json.load(file)

        training_features, _, test_features, test_labels = digits(training_size=20,
                                                                  test_size=10,
                                                                  n=2,
                                                                  plot_data=False)

        training_features_ref = np.concatenate(list(training_input_ref.values()))
        np.testing.assert_almost_equal(training_features_ref, training_features, 4)

        test_features_ref = np.concatenate(list(test_input_ref.values()))
        np.testing.assert_almost_equal(test_features_ref, test_features, 3)

<<<<<<< HEAD
        np.testing.assert_array_equal(test_labels.shape, (100, 10))
        np.testing.assert_array_equal(np.sum(test_labels, axis=0), np.array([10] * 10))
        np.testing.assert_array_equal(np.sum(test_labels, axis=1), np.ones(100))
=======
        np.testing.assert_almost_equal(ref_data["test_features"], test_features, 3)
        np.testing.assert_almost_equal(ref_data["test_labels"], test_labels, 4)
>>>>>>> eb2ac94... Disable Conan on CI Aer build (#45)


if __name__ == '__main__':
    unittest.main()
