# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Ad Hoc Data"""

from test import QiskitMachineLearningTestCase

import unittest
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
            training_size=20,
            test_size=10,
            n=num_features,
            gap=0.3,
            plot_data=False,
            one_hot=False,
        )
        np.testing.assert_array_equal(training_features.shape, (40, num_features))
        np.testing.assert_array_equal(training_labels.shape, (40,))
        np.testing.assert_array_almost_equal(test_labels, np.array([0] * 10 + [1] * 10))

        # Now one_hot=True
        _, _, _, test_labels_oh = ad_hoc_data(
            training_size=20,
            test_size=10,
            n=num_features,
            gap=0.3,
            plot_data=False,
            one_hot=True,
        )
        np.testing.assert_array_equal(test_labels_oh.shape, (20, 2))
        np.testing.assert_array_equal(test_labels_oh, np.array([[1, 0]] * 10 + [[0, 1]] * 10))

    def test_ref_data(self):
        """Tests ad hoc against known reference data"""
        input_file = self.get_resource_path("ad_hoc_ref.json", "datasets")
        with open(input_file, encoding="utf8") as file:
            ref_data = json.load(file)

        for seed in ref_data:
            algorithm_globals.random_seed = int(seed)
            (
                training_features,
                training_labels,
                test_features,
                test_labels,
            ) = ad_hoc_data(
                training_size=20,
                test_size=5,
                n=2,
                gap=0.3,
                plot_data=False,
                one_hot=False,
            )
            with self.subTest("Test training_features"):
                np.testing.assert_almost_equal(
                    ref_data[seed]["training_features"],
                    training_features,
                )
            with self.subTest("Test training_labels"):
                np.testing.assert_almost_equal(
                    ref_data[seed]["training_labels"],
                    training_labels,
                )
            with self.subTest("Test test_features"):
                np.testing.assert_almost_equal(
                    ref_data[seed]["test_features"],
                    test_features,
                )
            with self.subTest("Test test_labels"):
                np.testing.assert_almost_equal(
                    ref_data[seed]["test_labels"],
                    test_labels,
                )

    def test_entanglement_linear(self):
        """Test linear entanglement."""
        (
            training_features,
            training_labels,
            test_features,
            test_labels,
        ) = ad_hoc_data(
            training_size=10,
            test_size=5,
            n=2,
            plot_data=False,
            one_hot=False,
            entanglement="linear",
        )
        self.assertEqual(training_features.shape, (20, 2))
        self.assertEqual(training_labels.shape, (20,))
        self.assertEqual(test_features.shape, (10, 2))
        self.assertEqual(test_labels.shape, (10,))

    def test_entanglement_circular(self):
        """Test circular entanglement."""
        (
            training_features,
            training_labels,
            test_features,
            test_labels,
        ) = ad_hoc_data(
            training_size=10,
            test_size=5,
            n=2,
            plot_data=False,
            one_hot=False,
            entanglement="circular",
        )
        self.assertEqual(training_features.shape, (20, 2))
        self.assertEqual(training_labels.shape, (20,))
        self.assertEqual(test_features.shape, (10, 2))
        self.assertEqual(test_labels.shape, (10,))

    def test_entanglement_full(self):
        """Test full entanglement."""
        (
            training_features,
            training_labels,
            test_features,
            test_labels,
        ) = ad_hoc_data(
            training_size=10,
            test_size=5,
            n=2,
            plot_data=False,
            one_hot=False,
            entanglement="full",
        )
        self.assertEqual(training_features.shape, (20, 2))
        self.assertEqual(training_labels.shape, (20,))
        self.assertEqual(test_features.shape, (10, 2))
        self.assertEqual(test_labels.shape, (10,))

    def test_sampling_grid(self):
        """Test grid sampling method."""
        (
            training_features,
            training_labels,
            test_features,
            test_labels,
        ) = ad_hoc_data(
            training_size=10,
            test_size=5,
            n=2,
            sampling_method="grid",
            plot_data=False,
            one_hot=False,
        )
        self.assertEqual(training_features.shape, (20, 2))
        self.assertEqual(training_labels.shape, (20,))
        self.assertEqual(test_features.shape, (10, 2))
        self.assertEqual(test_labels.shape, (10,))

    def test_sampling_sobol(self):
        """Test Sobol sampling method."""
        (
            training_features,
            training_labels,
            test_features,
            test_labels,
        ) = ad_hoc_data(
            training_size=10,
            test_size=5,
            n=2,
            sampling_method="sobol",
            plot_data=False,
            one_hot=False,
        )
        self.assertEqual(training_features.shape, (20, 2))
        self.assertEqual(training_labels.shape, (20,))
        self.assertEqual(test_features.shape, (10, 2))
        self.assertEqual(test_labels.shape, (10,))

    def test_sampling_hypercube(self):
        """Test hypercube sampling with divisions parameter."""
        (
            training_features,
            training_labels,
            test_features,
            test_labels,
        ) = ad_hoc_data(
            training_size=10,
            test_size=5,
            n=2,
            sampling_method="hypercube",
            divisions=10,
            plot_data=False,
            one_hot=False,
        )
        self.assertEqual(training_features.shape, (20, 2))
        self.assertEqual(training_labels.shape, (20,))
        self.assertEqual(test_features.shape, (10, 2))
        self.assertEqual(test_labels.shape, (10,))

    def test_labelling_expectation(self):
        """Test expectation labelling method."""
        (
            training_features,
            training_labels,
            test_features,
            test_labels,
        ) = ad_hoc_data(
            training_size=10,
            test_size=5,
            n=2,
            plot_data=False,
            one_hot=False,
            labelling_method="expectation",
        )
        self.assertEqual(training_features.shape, (20, 2))
        self.assertEqual(training_labels.shape, (20,))
        self.assertEqual(test_features.shape, (10, 2))
        self.assertEqual(test_labels.shape, (10,))

    def test_labelling_measurement(self):
        """Test measurement labelling method."""
        (
            training_features,
            training_labels,
            test_features,
            test_labels,
        ) = ad_hoc_data(
            training_size=10,
            test_size=5,
            n=2,
            plot_data=False,
            one_hot=False,
            labelling_method="measurement",
        )
        self.assertEqual(training_features.shape, (20, 2))
        self.assertEqual(training_labels.shape, (20,))
        self.assertEqual(test_features.shape, (10, 2))
        self.assertEqual(test_labels.shape, (10,))

    def test_custom_class_labels(self):
        """Test custom class labels."""
        custom_labels = ["Class1", "Class2"]
        (
            _,
            training_labels,
            _,
            _,
        ) = ad_hoc_data(
            training_size=10,
            test_size=5,
            n=2,
            plot_data=False,
            one_hot=False,
            class_labels=custom_labels,
        )

        unique_labels = np.unique(training_labels)
        self.assertEqual(len(unique_labels), 2)
        for label in custom_labels:
            self.assertIn(label, unique_labels)

        # Test with one_hot=True
        (
            _,
            training_labels_onehot,
            _,
            test_labels_onehot,
        ) = ad_hoc_data(
            training_size=10,
            test_size=5,
            n=2,
            plot_data=False,
            one_hot=True,
            class_labels=custom_labels,
        )

        self.assertEqual(training_labels_onehot.shape, (20, 2))
        self.assertEqual(test_labels_onehot.shape, (10, 2))

    def test_include_sample_total(self):
        """Test include_sample_total parameter returns 5-tuple."""
        result = ad_hoc_data(
            training_size=10,
            test_size=5,
            n=2,
            plot_data=False,
            one_hot=False,
            include_sample_total=True,
        )
        self.assertEqual(len(result), 5)
        np.testing.assert_array_equal(result[4], np.array([30]))

    def test_higher_qubits(self):
        """Test with dimensions higher than 3 (n=4)."""
        (
            training_features,
            _,
            test_features,
            _,
        ) = ad_hoc_data(
            training_size=5,
            test_size=3,
            n=4,
            plot_data=False,
            one_hot=False,
            sampling_method="sobol",
        )
        self.assertEqual(training_features.shape, (10, 4))
        self.assertEqual(test_features.shape, (6, 4))

    def test_error_cases(self):
        """Test error cases in the new implementation."""

        # Test negative training_size
        with self.assertRaises(ValueError):
            ad_hoc_data(training_size=-1, test_size=5, n=2)

        # Test negative test_size
        with self.assertRaises(ValueError):
            ad_hoc_data(training_size=5, test_size=-1, n=2)

        # Test invalid n
        with self.assertRaises(ValueError):
            ad_hoc_data(training_size=5, test_size=5, n=0)

        # Test negative gap with expectation labelling
        with self.assertRaises(ValueError):
            ad_hoc_data(
                training_size=5,
                test_size=5,
                n=2,
                gap=-1,
                labelling_method="expectation",
            )

        # Test invalid entanglement
        with self.assertRaises(ValueError):
            ad_hoc_data(
                training_size=5,
                test_size=5,
                n=2,
                entanglement="invalid",
            )

        # Test invalid sampling method
        with self.assertRaises(ValueError):
            ad_hoc_data(
                training_size=5,
                test_size=5,
                n=2,
                sampling_method="invalid",
            )

        # Test hypercube without divisions
        with self.assertRaises(ValueError):
            ad_hoc_data(
                training_size=5,
                test_size=5,
                n=2,
                sampling_method="hypercube",
            )

        # Test invalid labelling method
        with self.assertRaises(ValueError):
            ad_hoc_data(
                training_size=5,
                test_size=5,
                n=2,
                labelling_method="invalid",
            )

        # Test grid sampling with n > 3
        with self.assertRaises(ValueError):
            ad_hoc_data(
                training_size=5,
                test_size=5,
                n=4,
                sampling_method="grid",
            )

    def test_hypercube_sampling_linear_entanglement(self):
        """Test hypercube sampling and linear entanglement."""
        (
            training_features,
            _,
            test_features,
            _,
        ) = ad_hoc_data(
            training_size=10,
            test_size=5,
            n=2,
            plot_data=False,
            one_hot=False,
            sampling_method="hypercube",
            divisions=12,
            entanglement="linear",
        )
        self.assertEqual(training_features.shape, (20, 2))
        self.assertEqual(test_features.shape, (10, 2))

    def test_custom_labels_circular_entanglement(self):
        """Test custom labels with circular entanglement."""
        custom_labels = ["Yes", "No"]
        (
            training_features,
            training_labels,
            test_features,
            _,
        ) = ad_hoc_data(
            training_size=8,
            test_size=4,
            n=3,
            plot_data=False,
            one_hot=False,
            entanglement="circular",
            class_labels=custom_labels,
        )
        self.assertEqual(training_features.shape, (16, 3))
        self.assertEqual(test_features.shape, (8, 3))
        unique_labels = np.unique(training_labels)
        self.assertIn("Yes", unique_labels)
        self.assertIn("No", unique_labels)

    def test_measurement_sobol_sampling(self):
        """Test custom labels with circular entanglement."""
        (
            training_features,
            _,
            test_features,
            _,
        ) = ad_hoc_data(
            training_size=8,
            test_size=4,
            n=3,
            plot_data=False,
            one_hot=False,
            labelling_method="measurement",
            sampling_method="sobol",
        )
        self.assertEqual(training_features.shape, (16, 3))
        self.assertEqual(test_features.shape, (8, 3))

    def test_expectation_labelling_with_gap(self):
        """Test expectation labelling with a non-zero gap."""
        (
            training_features,
            _,
            test_features,
            _,
        ) = ad_hoc_data(
            training_size=10,
            test_size=5,
            n=2,
            gap=0.5,
            plot_data=False,
            one_hot=False,
            labelling_method="expectation",
        )
        self.assertEqual(training_features.shape, (20, 2))
        self.assertEqual(test_features.shape, (10, 2))


if __name__ == "__main__":
    unittest.main()
