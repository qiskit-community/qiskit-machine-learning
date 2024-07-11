# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Ad Hoc Data """

import unittest

from test import QiskitMachineLearningTestCase

import json
import numpy as np
from ddt import ddt, unpack, idata

from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.datasets import ad_hoc_data


@ddt
class TestAdHocData(QiskitMachineLearningTestCase):
    """Ad Hoc Data tests."""

    @idata(
        ([2], [3]),
    )
    @unpack
    def test_ad_hoc_data(self, num_features):
        """Ad Hoc Data test."""

        training_features, training_labels, _, test_labels = ad_hoc_data(
            training_size=20, test_size=10, n=num_features, gap=0.3, plot_data=False, one_hot=False
        )
        np.testing.assert_array_equal(training_features.shape, (40, num_features))
        np.testing.assert_array_equal(training_labels.shape, (40,))
        np.testing.assert_array_almost_equal(
            test_labels, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )

        _, _, _, test_labels = ad_hoc_data(
            training_size=20, test_size=10, n=num_features, gap=0.3, plot_data=False, one_hot=True
        )

        np.testing.assert_array_equal(test_labels.shape, (20, 2))
        np.testing.assert_array_equal(
            test_labels,
            [
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
            ],
        )

    @idata(
        ([1], [4]),
    )
    @unpack
    def test_wrong_params(self, num_features):
        """Tests Ad Hoc Data with wrong parameters."""
        with self.assertRaises(ValueError):
            _, _, _, _ = ad_hoc_data(training_size=20, test_size=10, n=num_features, gap=0.3)

    def test_ref_data(self):
        """Tests ad hoc against known reference data"""
        input_file = self.get_resource_path("ad_hoc_ref.json", "datasets")
        with open(input_file, encoding="utf8") as file:
            ref_data = json.load(file)
        for seed in ref_data:
            algorithm_globals.random_seed = int(seed)
            training_features, training_labels, test_features, test_labels = ad_hoc_data(
                training_size=20, test_size=5, n=2, gap=0.3, plot_data=False, one_hot=False
            )
            with self.subTest("Test training_features"):
                np.testing.assert_almost_equal(
                    ref_data[seed]["training_features"], training_features
                )
            with self.subTest("Test training_labels"):
                np.testing.assert_almost_equal(ref_data[seed]["training_labels"], training_labels)
            with self.subTest("Test test_features"):
                np.testing.assert_almost_equal(ref_data[seed]["test_features"], test_features)
            with self.subTest("Test test_labels"):
                np.testing.assert_almost_equal(ref_data[seed]["test_labels"], test_labels)


if __name__ == "__main__":
    unittest.main()
