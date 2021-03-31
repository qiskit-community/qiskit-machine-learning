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

""" Test Iris """

import unittest
from test import QiskitMachineLearningTestCase
import json
import numpy as np
from qiskit_machine_learning.datasets import iris


class TestIris(QiskitMachineLearningTestCase):
    """Iris tests."""


    def test_iris(self):
        """Iris test."""

        # input_file = self.get_resource_path('sample_train.iris',
        #                                     'datasets')
        # with open(input_file) as file:
        #     sample_train_ref = json.load(file)

        input_file = self.get_resource_path('training_input.iris',
                                            'datasets')
        with open(input_file) as file:
            training_input_ref = json.load(file)

        input_file = self.get_resource_path('test_input.iris',
                                            'datasets')
        with open(input_file) as file:
            test_input_ref = json.load(file)

        training_features, _, test_features, test_labels = iris(training_size=20,
                                                                test_size=3,
                                                                n=2,
                                                                plot_data=False)

        training_features_ref = np.concatenate(list(training_input_ref.values()))
        np.testing.assert_almost_equal(training_features_ref, training_features, 3)

        test_features_ref = np.concatenate([x for x in list(test_input_ref.values()) if len(x) > 0])
        np.testing.assert_almost_equal(test_features_ref, test_features, 3)
        #
        np.testing.assert_array_equal(test_labels.shape, (1, 3))

        np.testing.assert_array_equal(np.sum(test_labels, axis=1), np.ones(1))


if __name__ == '__main__':
    unittest.main()

