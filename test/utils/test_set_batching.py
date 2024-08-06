# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test setting default batch sizes for the optimizers."""

from test import QiskitAlgorithmsTestCase

from unittest.mock import Mock
from qiskit_machine_learning.optimizers import Optimizer, SPSA
from qiskit_machine_learning.utils.set_batching import _set_default_batchsize


class TestSetDefaultBatchsize(QiskitAlgorithmsTestCase):
    """Test the ``_set_default_batchsize`` utility function."""

    def setUp(self):
        super().setUp()
        self.spsa_optimizer = Mock(spec=SPSA)
        self.generic_optimizer = Mock(spec=Optimizer)

    def test_set_default_batchsize_for_spsa_with_none(self):
        """Test setting default batchsize for SPSA when _max_evals_grouped is None."""
        self.spsa_optimizer._max_evals_grouped = None
        self.spsa_optimizer.set_max_evals_grouped = Mock()

        updated = _set_default_batchsize(self.spsa_optimizer)

        self.spsa_optimizer.set_max_evals_grouped.assert_called_once_with(50)
        self.assertTrue(updated)

    def test_set_default_batchsize_for_spsa_with_value(self):
        """Test setting default batchsize for SPSA when _max_evals_grouped is already set."""
        self.spsa_optimizer._max_evals_grouped = 10

        updated = _set_default_batchsize(self.spsa_optimizer)

        self.spsa_optimizer.set_max_evals_grouped.assert_not_called()
        self.assertFalse(updated)

    def test_set_default_batchsize_for_generic_optimizer(self):
        """Test setting default batchsize for a non-SPSA optimizer."""
        updated = _set_default_batchsize(self.generic_optimizer)

        self.assertFalse(updated)
