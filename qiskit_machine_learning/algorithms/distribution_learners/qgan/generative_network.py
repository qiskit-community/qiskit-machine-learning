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

""" Generative Quantum and Classical Neural Networks."""

from typing import Optional, List, Tuple
from abc import ABC, abstractmethod
import numpy as np
from qiskit.utils import QuantumInstance


class GenerativeNetwork(ABC):
    """
    Base class for generative Quantum and Classical Neural Networks.

    This method should initialize the module, but
    raise an exception if a required component of the module is not available.
    """

    @abstractmethod
    def __init__(self):
        super().__init__()
        self._num_parameters = 0
        self._num_qubits = 0
        self._bounds = []

    @property
    @abstractmethod
    def parameter_values(self):
        """
        Get parameter values from the generator

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()

    @abstractmethod
    def get_output(
        self,
        quantum_instance: QuantumInstance,
        params: Optional[np.ndarray] = None,
        shots: Optional[int] = None,
    ) -> Tuple[List, List]:
        """
        Apply quantum/classical neural network to given input and get the respective output

        Args:
            quantum_instance: Quantum Instance, used to run the generator circuit.
            params: parameters which should be used to run the generator,
                if None use self._params
            shots: if not None use a number of shots that is different from the number
                set in quantum_instance

        Returns:
            Neural network output

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()

    @abstractmethod
    def loss(self):
        """
        Loss function used for optimization
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, quantum_instance=None, shots=None):
        """
        Perform one training step w.r.t to the generator's parameters

        Args:
            quantum_instance (QuantumInstance): used to run generator network.
               Ignored for a classical network.
            shots (int): Number of shots for hardware or qasm execution.
                Ignored for classical network

        Returns:
            dict: generator loss and updated parameters.

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()
