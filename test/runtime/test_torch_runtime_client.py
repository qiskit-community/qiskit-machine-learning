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
from test import QiskitMachineLearningTestCase, requires_extra_library

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.runtime import TorchRuntimeClient
from .fake_torchruntime import FakeTorchInferRuntimeProvider, FakeTorchTrainerRuntimeProvider

try:
    from torch import is_tensor, Tensor
    from torch.nn import MSELoss
    from torch.optim import Adam
    from torch.utils.data import Dataset
except ImportError:

    class Dataset:  # type: ignore
        """Empty Dataset class
        Replacement if torch.utils.data.Dataset is not present.
        """

        pass

    class Tensor:  # type: ignore
        """Empty Tensor class
        Replacement if torch.Tensor is not present.
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

        if is_tensor(idx):
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
        # construct a simple qnn for unit tests
        param_x = Parameter("x")
        feature_map = QuantumCircuit(1, name="fm")
        feature_map.ry(param_x, 0)
        param_y = Parameter("y")
        ansatz = QuantumCircuit(1, name="vf")
        ansatz.ry(param_y, 0)
        self._qnn = TwoLayerQNN(1, feature_map, ansatz)

    @requires_extra_library
    def test_fit(self):
        """Test for fit"""
        try:
            from torch.utils.data import DataLoader
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="Pytorch",
                name="TorchConnector",
                pip_install="pip install 'qiskit-machine-learning[torch]'",
            ) from ex
        train_loader = DataLoader(TorchDataset([1], [1]), batch_size=1, shuffle=False)
        model = TorchConnector(self._qnn, [1])
        backend = QasmSimulatorPy()
        optimizer = Adam(model.parameters(), lr=0.1)
        loss_func = MSELoss(reduction="sum")
        torch_program = TorchRuntimeClient(
            model=model,
            optimizer=optimizer,
            loss_func=loss_func,
            provider=self._trainer_provider,
            epochs=1,
            backend=backend,
            shots=1024,
        )
        torch_program.fit(train_loader)

    @requires_extra_library
    def test_predict(self):
        """Test for predict"""
        try:
            from torch.utils.data import DataLoader
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="Pytorch",
                name="TorchConnector",
                pip_install="pip install 'qiskit-machine-learning[torch]'",
            ) from ex
        train_loader = DataLoader(TorchDataset([1], [1]), batch_size=1, shuffle=False)
        model = TorchConnector(self._qnn, [1])
        backend = QasmSimulatorPy()
        optimizer = Adam(model.parameters(), lr=0.1)
        loss_func = MSELoss(reduction="sum")
        torch_program = TorchRuntimeClient(
            model=model,
            optimizer=optimizer,
            loss_func=loss_func,
            provider=self._infer_provider,
            epochs=1,
            backend=backend,
            shots=1024,
        )
        torch_program.predict(train_loader)

    @requires_extra_library
    def test_score(self):
        """Test for score"""
        try:
            from torch.utils.data import DataLoader
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="Pytorch",
                name="TorchConnector",
                pip_install="pip install 'qiskit-machine-learning[torch]'",
            ) from ex
        train_loader = DataLoader(TorchDataset([1], [1]), batch_size=1, shuffle=False)
        model = TorchConnector(self._qnn, [1])
        backend = QasmSimulatorPy()
        optimizer = Adam(model.parameters(), lr=0.1)
        loss_func = MSELoss(reduction="sum")
        torch_program = TorchRuntimeClient(
            model=model,
            optimizer=optimizer,
            loss_func=loss_func,
            provider=self._infer_provider,
            epochs=1,
            backend=backend,
            shots=1024,
        )
        torch_program.score(train_loader, score_func="Regression")


if __name__ == "__main__":
    unittest.main()
