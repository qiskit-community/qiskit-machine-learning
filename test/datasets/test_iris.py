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

        input_file = self.get_resource_path("iris_ref.json", "datasets")
        with open(input_file) as file:
            ref_data = json.load(file)

        training_features, training_labels, test_features, test_labels = iris(
            training_size=20, test_size=3, n=2, plot_data=False
        )

        np.testing.assert_almost_equal(ref_data["training_features"], training_features)
        np.testing.assert_almost_equal(ref_data["training_labels"], training_labels)

        np.testing.assert_almost_equal(ref_data["test_features"], test_features)
        np.testing.assert_almost_equal(ref_data["test_labels"], test_labels)


if __name__ == "__main__":
    unittest.main()
