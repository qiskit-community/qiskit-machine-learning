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

"""Test Torch Base."""

import unittest
import builtins
from abc import ABC, abstractmethod
from qiskit_machine_learning.utils import algorithm_globals
import qiskit_machine_learning.optionals as _optionals


class TestTorch(ABC):
    """Torch Base Tests."""

    def __init__(self):
        self._device = None
        self._sv_quantum_instance = None
        self._qasm_quantum_instance = None

    @unittest.skipUnless(_optionals.HAS_TORCH, "PyTorch not available.")
    def setup_test(self):
        """Base setup."""

        algorithm_globals.random_seed = 12345
        import torch

        torch.manual_seed(algorithm_globals.random_seed)

    @abstractmethod
    def subTest(self, msg, **kwargs):
        # pylint: disable=invalid-name
        """Sub test."""
        raise builtins.Exception("Abstract method")

    # pylint: disable=too-many-positional-arguments
    @abstractmethod
    def assertAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        """Assert almost equal."""
        raise builtins.Exception("Abstract method")

    @abstractmethod
    def assertEqual(self, first, second, msg=None):
        """Assert equal."""
        raise builtins.Exception("Abstract method")

    @abstractmethod
    def assertTrue(self, expr, msg=None):
        """Assert true."""
        raise builtins.Exception("Abstract method")

    @abstractmethod
    def assertFalse(self, expr, msg=None):
        """assert False"""
        raise builtins.Exception("Abstract method")

    @abstractmethod
    def skipTest(self, reason):  # pylint: disable=invalid-name
        """Skip test."""
        raise builtins.Exception("Abstract method")

    @abstractmethod
    def assertLogs(self, logger=None, level=None):
        """Assert logs."""
        raise builtins.Exception("Abstract method")

    @abstractmethod
    def assertListEqual(self, list1, list2, msg=None):
        """Assert list equal."""
        raise builtins.Exception("Abstract method")

    @abstractmethod
    def assertRaises(self, expected_exception):
        """Assert raises an exception."""
        raise builtins.Exception("Abstract method")
