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
"""GPU based tests of the PyTorch connector."""

import unittest
from test import QiskitMachineLearningTestCase, gpu
from test.connectors.test_torch_connector import TestTorchConnector


class TestGPUTorchConnector(QiskitMachineLearningTestCase, TestTorchConnector):
    """GPU based tests of the PyTorch connector."""

    @gpu
    def setUp(self):
        super().setup_test()
        super().setUp()

        import torch

        if not torch.cuda.is_available():
            # raise test exception
            self.skipTest("CUDA is not available")
        else:
            self._device = torch.device("cuda")

    def tearDown(self) -> None:
        super().tearDown()
        super().tear_down()


if __name__ == "__main__":
    unittest.main()
