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

""" Test Ad Hoc Data """

from test.ml import QiskitMLTestCase
import numpy as np
from qiskit.ml.datasets import ad_hoc_data
from qiskit.aqua.utils import split_dataset_to_data_and_labels


class TestAdHocData(QiskitMLTestCase):
    """Ad Hoc Data tests."""

    def test_ad_hoc_data(self):
        """Ad Hoc Data test."""

        _, _, test_input, class_labels = ad_hoc_data(training_size=20,
                                                     test_size=10,
                                                     n=2,
                                                     gap=0.3,
                                                     plot_data=False)

        np.testing.assert_array_equal(class_labels, ['A', 'B'])

        datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)
        np.testing.assert_array_equal(datapoints[1].tolist(),
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertDictEqual(class_to_label, {'A': 0, 'B': 1})
