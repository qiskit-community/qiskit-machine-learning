# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Torch Connector 2."""

import unittest
from test import QiskitMachineLearningTestCase
from test.connectors.test_torch_models import TestTorchModels

from qiskit.utils import optionals
import qiskit_machine_learning.optionals as _optionals


class TestHybridTorchModels(QiskitMachineLearningTestCase, TestTorchModels):
    """Hybrid model tests for Torch Connector."""

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    @unittest.skipIf(not _optionals.HAS_TORCH, "PyTorch not available.")
    def setUp(self):
        super().setup_test()
        super().setUp()

    def _get_device(self):
        import torch

        return torch.device("cpu")


if __name__ == "__main__":
    unittest.main()
