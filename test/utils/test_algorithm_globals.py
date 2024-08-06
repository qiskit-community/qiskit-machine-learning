# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test QiskitAlgorithmGlobals."""

from test import QiskitAlgorithmsTestCase
from unittest.mock import patch
import numpy as np

from qiskit_machine_learning.utils.algorithm_globals import QiskitAlgorithmGlobals


class TestQiskitAlgorithmGlobals(QiskitAlgorithmsTestCase):
    """Test the QiskitAlgorithmGlobals class."""

    def setUp(self):
        super().setUp()
        self.algorithm_globals = QiskitAlgorithmGlobals()

    @patch("qiskit.utils.algorithm_globals", create=True)
    def test_random_seed_getter_qiskit(self, mock_qiskit_globals):
        """Test random_seed getter when qiskit_machine_learning.utils.algorithm_globals
        is available."""
        mock_qiskit_globals.random_seed = 42

        seed = self.algorithm_globals.random_seed

        self.assertEqual(seed, 42)

    def test_random_seed_getter_local(self):
        """Test random_seed getter when qiskit_machine_learning.utils.algorithm_globals
        is not available."""
        self.algorithm_globals._random_seed = 24

        seed = self.algorithm_globals.random_seed

        self.assertEqual(seed, 24)

    @patch("qiskit.utils.algorithm_globals", create=True)
    def test_random_seed_setter_qiskit(self, mock_qiskit_globals):
        """Test random_seed setter when qiskit_machine_learning.utils.algorithm_globals
        is available."""
        self.algorithm_globals.random_seed = 15

        self.assertEqual(mock_qiskit_globals.random_seed, 15)
        self.assertEqual(self.algorithm_globals._random_seed, 15)

    def test_random_seed_setter_local(self):
        """Test random_seed setter when qiskit_machine_learning.utils.algorithm_globals
        is not available."""
        self.algorithm_globals.random_seed = 7

        self.assertEqual(self.algorithm_globals._random_seed, 7)
        self.assertIsNone(self.algorithm_globals._random)

    def test_random_property_local(self):
        """Test random property when qiskit_machine_learning.utils.algorithm_globals is not available."""
        self.algorithm_globals.random_seed = 5
        rng = self.algorithm_globals.random

        self.assertEqual(self.algorithm_globals._random_seed, 5)
        self.assertIsInstance(rng, np.random.Generator)
        self.assertEqual(rng.bit_generator._seed_seq.entropy, 5)

    def test_random_property_local_no_seed(self):
        """Test random property when qiskit_machine_learning.utils.algorithm_globals
        is not available and seed is None."""
        rng = self.algorithm_globals.random

        self.assertIsNone(self.algorithm_globals._random_seed)
        self.assertIsInstance(rng, np.random.Generator)
