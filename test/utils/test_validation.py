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

"""Test validation functions."""

from test import QiskitAlgorithmsTestCase
from qiskit_machine_learning.utils.validation import (
    validate_in_set,
    validate_min,
    validate_min_exclusive,
    validate_max,
    validate_max_exclusive,
    validate_range,
    validate_range_exclusive,
    validate_range_exclusive_min,
    validate_range_exclusive_max,
)


class TestValidationFunctions(QiskitAlgorithmsTestCase):
    """Test the validation functions."""

    def test_validate_in_set_valid(self):
        """Test validate_in_set with valid value."""
        validate_in_set("param", "a", {"a", "b", "c"})  # Should not raise

    def test_validate_in_set_invalid(self):
        """Test validate_in_set with invalid value."""
        with self.assertRaises(ValueError):
            validate_in_set("param", "d", {"a", "b", "c"})

    def test_validate_min_valid(self):
        """Test validate_min with valid value."""
        validate_min("param", 5, 1)  # Should not raise

    def test_validate_min_invalid(self):
        """Test validate_min with invalid value."""
        with self.assertRaises(ValueError):
            validate_min("param", 0, 1)

    def test_validate_min_exclusive_valid(self):
        """Test validate_min_exclusive with valid value."""
        validate_min_exclusive("param", 5, 1)  # Should not raise

    def test_validate_min_exclusive_invalid(self):
        """Test validate_min_exclusive with invalid value."""
        with self.assertRaises(ValueError):
            validate_min_exclusive("param", 1, 1)

    def test_validate_max_valid(self):
        """Test validate_max with valid value."""
        validate_max("param", 5, 10)  # Should not raise

    def test_validate_max_invalid(self):
        """Test validate_max with invalid value."""
        with self.assertRaises(ValueError):
            validate_max("param", 15, 10)

    def test_validate_max_exclusive_valid(self):
        """Test validate_max_exclusive with valid value."""
        validate_max_exclusive("param", 5, 10)  # Should not raise

    def test_validate_max_exclusive_invalid(self):
        """Test validate_max_exclusive with invalid value."""
        with self.assertRaises(ValueError):
            validate_max_exclusive("param", 10, 10)

    def test_validate_range_valid(self):
        """Test validate_range with valid value."""
        validate_range("param", 5, 1, 10)  # Should not raise

    def test_validate_range_invalid(self):
        """Test validate_range with invalid value."""
        with self.assertRaises(ValueError):
            validate_range("param", 0, 1, 10)
        with self.assertRaises(ValueError):
            validate_range("param", 15, 1, 10)

    def test_validate_range_exclusive_valid(self):
        """Test validate_range_exclusive with valid value."""
        validate_range_exclusive("param", 5, 1, 10)  # Should not raise

    def test_validate_range_exclusive_invalid(self):
        """Test validate_range_exclusive with invalid value."""
        with self.assertRaises(ValueError):
            validate_range_exclusive("param", 1, 1, 10)
        with self.assertRaises(ValueError):
            validate_range_exclusive("param", 10, 1, 10)

    def test_validate_range_exclusive_min_valid(self):
        """Test validate_range_exclusive_min with valid value."""
        validate_range_exclusive_min("param", 5, 1, 10)  # Should not raise

    def test_validate_range_exclusive_min_invalid(self):
        """Test validate_range_exclusive_min with invalid value."""
        with self.assertRaises(ValueError):
            validate_range_exclusive_min("param", 1, 1, 10)
        with self.assertRaises(ValueError):
            validate_range_exclusive_min("param", 11, 1, 10)

    def test_validate_range_exclusive_max_valid(self):
        """Test validate_range_exclusive_max with valid value."""
        validate_range_exclusive_max("param", 5, 1, 10)  # Should not raise

    def test_validate_range_exclusive_max_invalid(self):
        """Test validate_range_exclusive_max with invalid value."""
        with self.assertRaises(ValueError):
            validate_range_exclusive_max("param", 0, 1, 10)
        with self.assertRaises(ValueError):
            validate_range_exclusive_max("param", 10, 1, 10)
