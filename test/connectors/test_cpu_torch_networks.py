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
"""CPU based tests of hybrid PyTorch networks."""

import unittest
from test import QiskitMachineLearningTestCase
from test.connectors.test_torch_networks import TestTorchNetworks

from qiskit.utils import optionals
import qiskit_machine_learning.optionals as _optionals


class TestCPUTorchNetworks(QiskitMachineLearningTestCase, TestTorchNetworks):
    """CPU based tests of hybrid PyTorch networks."""

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    @unittest.skipIf(not _optionals.HAS_TORCH, "PyTorch not available.")
    def setUp(self):
        super().setup_test()
        super().setUp()

        import torch

        self._device = torch.device("cpu")


if __name__ == "__main__":
    unittest.main()
