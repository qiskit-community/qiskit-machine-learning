# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Wine """

from test.ml import QiskitMLTestCase
import json
import numpy as np
from qiskit.ml.datasets import wine
from qiskit.aqua.utils import split_dataset_to_data_and_labels


class TestWine(QiskitMLTestCase):
    """Wine tests."""

    def test_wine(self):
        """Wine test."""

        input_file = self.get_resource_path('sample_train.wine')
        with open(input_file) as file:
            sample_train_ref = json.load(file)

        input_file = self.get_resource_path('training_input.wine')
        with open(input_file) as file:
            training_input_ref = json.load(file)

        input_file = self.get_resource_path('test_input.wine')
        with open(input_file) as file:
            test_input_ref = json.load(file)

        sample_train, training_input, test_input, class_labels = wine(training_size=20,
                                                                      test_size=10,
                                                                      n=2,
                                                                      plot_data=False)

        np.testing.assert_allclose(sample_train.tolist(), sample_train_ref, rtol=1e-04)
        for key, _ in training_input.items():
            np.testing.assert_allclose(training_input[key].tolist(),
                                       training_input_ref[key], rtol=1e-04)
        for key, _ in test_input.items():
            np.testing.assert_allclose(test_input[key].tolist(), test_input_ref[key], rtol=1e-04)
        np.testing.assert_array_equal(class_labels, list(training_input.keys()))

        datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)
        np.testing.assert_array_equal(datapoints[1], [0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        self.assertDictEqual(class_to_label, {'A': 0, 'B': 1, 'C': 2})
