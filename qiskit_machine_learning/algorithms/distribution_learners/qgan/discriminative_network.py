# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Discriminative Quantum or Classical Neural Networks."""

from typing import List, Iterable, Optional, Dict
from abc import ABC, abstractmethod

import numpy as np

from qiskit.utils import QuantumInstance


class DiscriminativeNetwork(ABC):
    """
    Base class for discriminative Quantum or Classical Neural Networks.

    This method should initialize the module but
    raise an exception if a required component of the module is not available.
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self._num_parameters = 0
        self._num_qubits = 0
        self._bounds = []  # type: List[object]

    @abstractmethod
    def set_seed(self, seed):
        """
        Set seed.

        Args:
            seed (int): seed

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()

    @abstractmethod
    def get_label(self, x: Iterable):
        """
        Apply quantum/classical neural network to the given input sample and compute
        the respective data label

        Args:
            x (Discriminator): input, i.e. data sample.

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()

    @abstractmethod
    def save_model(self, snapshot_dir: str):
        """
        Save discriminator model

        Args:
            snapshot_dir: Directory to save the model

        Raises:
            NotImplementedError: not implemented

        """
        raise NotImplementedError()

    @abstractmethod
    def loss(self, x: Iterable, y: Iterable, weights: Optional[np.ndarray] = None):
        """
        Loss function used for optimization

        Args:
            x: output.
            y: the data point
            weights: Data weights.

        Returns:
            Loss w.r.t to the generated data points.

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()

    @abstractmethod
    def train(
        self,
        data: Iterable,
        weights: Iterable,
        penalty: bool = False,
        quantum_instance: Optional[QuantumInstance] = None,
        shots: Optional[int] = None,
    ) -> Dict:
        """
        Perform one training step w.r.t to the discriminator's parameters

        Args:
            data: Data batch.
            weights: Data sample weights.
            penalty: Indicate whether or not penalty function
               is applied to the loss function. Ignored if no penalty function defined.
            quantum_instance (QuantumInstance): used to run Quantum network.
               Ignored for a classical network.
            shots: Number of shots for hardware or qasm execution.
                Ignored for classical network

        Returns:
            dict: with discriminator loss and updated parameters.

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()
