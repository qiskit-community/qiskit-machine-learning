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

""" Test SVM Classical Multiclass """

from test.ml.common import QiskitMLTestCase
import numpy as np
from qiskit.ml.datasets import wine
from qiskit.aqua.algorithms import SVM_Classical
from qiskit.aqua.components.multiclass_extensions import (OneAgainstRest,
                                                          AllPairs,
                                                          ErrorCorrectingCode)
from qiskit.aqua.algorithms.classical.svm import _RBF_SVC_Estimator


class TestSVMClassicalMulticlass(QiskitMLTestCase):
    """SVM Classical Multiclass tests."""

    def test_svm_classical_multiclass(self):
        """SVM Classical Multiclass test."""

        _, training_input, test_input, _ = wine(training_size=20,
                                                test_size=10,
                                                n=2,  # dimension of each data point
                                                plot_data=False)

        temp = [test_input[k] for k in test_input]
        total_array = np.concatenate(temp)

        extensions = [OneAgainstRest(_RBF_SVC_Estimator),
                      AllPairs(_RBF_SVC_Estimator),
                      ErrorCorrectingCode(_RBF_SVC_Estimator, code_size=5)]

        for extension in extensions:
            result = SVM_Classical(training_input,
                                   test_input,
                                   total_array,
                                   multiclass_extension=extension).run()
            self.assertAlmostEqual(result['testing_accuracy'], 1)
            np.testing.assert_array_equal(result['predicted_labels'],
                                          [0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
            np.testing.assert_array_equal(result['predicted_classes'],
                                          ['A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C'])
