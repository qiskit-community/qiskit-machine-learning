# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Discriminative Quantum or Classical Neural Networks."""

from typing import List
from abc import ABC, abstractmethod


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
        self._bounds = list()  # type: List[object]

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
    def get_label(self, x):
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
    def loss(self, x, y, weights=None):
        """
        Loss function used for optimization

        Args:
            x (Discriminator): output.
            y (Label): the data point
            weights (numpy.ndarray): Data weights.

        Returns:
            Loss w.r.t to the generated data points.

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, data, weights, penalty=False, quantum_instance=None, shots=None):
        """
        Perform one training step w.r.t to the discriminator's parameters

        Args:
            data (numpy.ndarray): Data batch.
            weights (numpy.ndarray): Data sample weights.
            penalty (bool): Indicate whether or not penalty function
               is applied to the loss function. Ignored if no penalty function defined.
            quantum_instance (QuantumInstance): used to run Quantum network.
               Ignored for a classical network.
            shots (int): Number of shots for hardware or qasm execution.
                Ignored for classical network

        Returns:
            dict: with Discriminator loss and updated parameters.

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()
