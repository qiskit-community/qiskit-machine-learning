# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for TorchProgram."""

import unittest
from test import QiskitMachineLearningTestCase

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.runtime import TorchRuntimeClient

from .fake_torchruntime import FakeTorchInferRuntimeProvider, FakeTorchTrainerRuntimeProvider

try:
    from torch import Tensor
    from torch.nn import MSELoss
    from torch.optim import Adam
    from torch.utils.data import DataLoader, Dataset
except ImportError:

    class Dataset:  # type: ignore
        """Empty Dataset class
        Replacement if torch.utils.data.Dataset is not present.
        """

        pass


class TorchDataset(Dataset):
    """Map-style dataset"""

    def __init__(self, x, y):
        # pylint: disable=W0231
        self.x = Tensor(x).float()
        self.y = Tensor(y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        import torch

        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_i = self.x[idx]
        y_i = self.y[idx]

        return x_i, y_i


class TestTorchRuntimeClient(QiskitMachineLearningTestCase):
    """Test the Torch program."""

    def setUp(self):
        super().setUp()
        self._trainer_provider = FakeTorchTrainerRuntimeProvider()
        self._infer_provider = FakeTorchInferRuntimeProvider()
        self._train_set = TorchDataset([1], [1])
        self._train_loader = DataLoader(self._train_set, batch_size=1, shuffle=False)
        # construct a simple qnn for unit tests
        param_x = Parameter("x")
        feature_map = QuantumCircuit(1, name="fm")
        feature_map.ry(param_x, 0)
        param_y = Parameter("y")
        ansatz = QuantumCircuit(1, name="vf")
        ansatz.ry(param_y, 0)
        qnn = TwoLayerQNN(1, feature_map, ansatz)
        self._model = TorchConnector(qnn, [1])

    def test_fit(self):
        """Test for fit"""
        backend = QasmSimulatorPy()
        optimizer = Adam(self._model.parameters(), lr=0.1)
        loss_func = MSELoss(reduction="sum")
        torch_program = TorchRuntimeClient(
            model=self._model,
            optimizer=optimizer,
            loss_func=loss_func,
            provider=self._trainer_provider,
            epochs=1,
            backend=backend,
            shots=1024,
        )
        torch_program.fit(self._train_loader)

    def test_predict(self):
        """Test for predict"""
        backend = QasmSimulatorPy()
        optimizer = Adam(self._model.parameters(), lr=0.1)
        loss_func = MSELoss(reduction="sum")
        torch_program = TorchRuntimeClient(
            model=self._model,
            optimizer=optimizer,
            loss_func=loss_func,
            provider=self._infer_provider,
            epochs=1,
            backend=backend,
            shots=1024,
        )
        torch_program.predict(self._train_loader)

    def test_score(self):
        """Test for score"""
        backend = QasmSimulatorPy()
        optimizer = Adam(self._model.parameters(), lr=0.1)
        loss_func = MSELoss(reduction="sum")
        torch_program = TorchRuntimeClient(
            model=self._model,
            optimizer=optimizer,
            loss_func=loss_func,
            provider=self._infer_provider,
            epochs=1,
            backend=backend,
            shots=1024,
        )
        torch_program.score(self._train_loader, score_func="Regression")


if __name__ == "__main__":
    unittest.main()
