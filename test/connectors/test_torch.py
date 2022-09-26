# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
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
from abc import ABC, abstractmethod
from qiskit.utils import QuantumInstance, algorithm_globals, optionals
import qiskit_machine_learning.optionals as _optionals


class TestTorch(ABC):
    """Torch Base Tests."""

    def __init__(self):
        self._device = None
        self._sv_quantum_instance = None
        self._qasm_quantum_instance = None

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    @unittest.skipUnless(_optionals.HAS_TORCH, "PyTorch not available.")
    def setup_test(self):
        """Base setup."""
        algorithm_globals.random_seed = 12345
        # specify quantum instances
        from qiskit_aer import Aer, AerSimulator

        self._sv_quantum_instance = QuantumInstance(
            Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        # pylint: disable=no-member
        self._qasm_quantum_instance = QuantumInstance(
            AerSimulator(),
            shots=100,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        import torch

        torch.manual_seed(algorithm_globals.random_seed)

    @abstractmethod
    def subTest(self, msg, **kwargs):
        # pylint: disable=invalid-name
        """Sub test."""
        raise Exception("Abstract method")

    @abstractmethod
    def assertAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        """Assert almost equal."""
        raise Exception("Abstract method")

    @abstractmethod
    def assertEqual(self, first, second, msg=None):
        """Assert equal."""
        raise Exception("Abstract method")

    @abstractmethod
    def assertTrue(self, expr, msg=None):
        """Assert true."""
        raise Exception("Abstract method")

    @abstractmethod
    def assertFalse(self, expr, msg=None):
        """assert False"""
        raise Exception("Abstract method")

    @abstractmethod
    def skipTest(self, reason):  # pylint: disable=invalid-name
        """Skip test."""
        raise Exception("Abstract method")

    @abstractmethod
    def assertLogs(self, logger=None, level=None):
        """Assert logs."""
        raise Exception("Abstract method")

    @abstractmethod
    def assertListEqual(self, list1, list2, msg=None):
        """Assert list equal."""
        raise Exception("Abstract method")
