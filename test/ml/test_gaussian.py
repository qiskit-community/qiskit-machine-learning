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

""" Test Gaussian """

from test.ml import QiskitMLTestCase
import numpy as np
from qiskit.ml.datasets import gaussian


class TestGaussian(QiskitMLTestCase):
    """Gaussian tests."""

    def test_gaussian(self):
        """Gaussian test."""

        _, _, _, class_labels = gaussian(training_size=20,
                                         test_size=10,
                                         n=2,
                                         plot_data=False)
        np.testing.assert_array_equal(class_labels, ['A', 'B'])
