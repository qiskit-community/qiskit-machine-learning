# -*- coding: utf-8 -*-

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

""" Test SVM Classical """

from test.ml.common import QiskitMLTestCase
from qiskit.ml.datasets import ad_hoc_data, breast_cancer
from qiskit.aqua.utils import split_dataset_to_data_and_labels
from qiskit.aqua.algorithms import SVM_Classical


class TestSVMClassical(QiskitMLTestCase):
    """SVM Classical tests."""

    def test_svm_classical(self):
        """SVM Classical test."""

        _, training_input, test_input, _ = ad_hoc_data(training_size=20,
                                                       test_size=10,
                                                       n=2,  # dimension of each data point
                                                       gap=0.3,
                                                       plot_data=False)
        datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)
        self.assertEqual(class_to_label, {'A': 0, 'B': 1})

        result = SVM_Classical(training_input, test_input, datapoints[0]).run()

        self.assertAlmostEqual(result['testing_accuracy'], 1, delta=0.5)

        _, training_input, test_input, _ = breast_cancer(training_size=20,
                                                         test_size=10,
                                                         n=2,
                                                         plot_data=False)

        datapoints, _ = split_dataset_to_data_and_labels(test_input)
        result = SVM_Classical(training_input, test_input, datapoints[0]).run()
        self.log.debug(result['testing_accuracy'])
        self.assertGreaterEqual(result['testing_accuracy'], 0.5)
