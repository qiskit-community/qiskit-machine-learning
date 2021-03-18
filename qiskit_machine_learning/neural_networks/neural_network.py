# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A Neural Network abstract class for all (quantum) neural networks within Qiskit's
machine learning module."""


from abc import ABC, abstractmethod
from typing import Tuple, Union, List, Optional, Dict

import numpy as np

from ..exceptions import QiskitMachineLearningError


class NeuralNetwork(ABC):
    """Abstract Neural Network class providing forward and backward pass and handling
    batched inputs. This is to be implemented by other (quantum) neural networks.
    """

    def __init__(self, num_inputs: int, num_weights: int,
                 output_shape: Union[int, Tuple[int, ...]]) -> None:
        """Initializes the Neural Network.
        Args:
            num_inputs: The number of input features.
            num_weights: The number of trainable weights.
            output_shape: The shape of the output.
        Raises:
            QiskitMachineLearningError: Invalid parameter values.
        """
        if num_inputs < 0:
            raise QiskitMachineLearningError('Number of inputs cannot be negative!')
        self._num_inputs = num_inputs

        if num_weights < 0:
            raise QiskitMachineLearningError('Number of weights cannot be negative!')
        self._num_weights = num_weights

        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        if not np.all([s > 0 for s in output_shape]):
            raise QiskitMachineLearningError('Invalid output shape, all components must be > 0!')
        self._output_shape = output_shape

    @property
    def num_inputs(self) -> int:
        """Returns the number of input features."""
        return self._num_inputs

    @property
    def num_weights(self) -> int:
        """Returns the number of trainable weights."""
        return self._num_weights

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Returns the output shape."""
        return self._output_shape

    def _validate_input(self, input_data: Optional[Union[List[float], np.ndarray, float]]):
        if input_data is None:
            return None
        input_ = np.array(input_data)
        shape = input_data.shape
        if shape[-1] != self._num_inputs:
            raise QiskitMachineLearningError(f"Input data has incorrect shape, last dimension "
                                             f"is not equal to the number of inputs: "
                                             f"{self._num_inputs}, but got: {shape[-1]}.")
        if len(shape) == 1:
            # add empty dimension for samples (batch dimension)
            input_ = input_.reshape((1, -1))
        elif len(shape) > 2:
            # flatten higher dimensions, keep num_inputs as a last dimension
            input_ = input_.reshape((np.product(input_.shape[:-1]), -1))

        return input_, shape

    def _validate_weights(self, weights: Optional[Union[List[float], np.ndarray, float]]):
        if weights is None:
            return None
        weights_ = np.array(weights)
        return weights_.reshape(self.num_weights)

    def forward(self, input_data: Optional[Union[List[float], np.ndarray, float]],
                weights: Optional[Union[List[float], np.ndarray, float]]
                ) -> Union[np.ndarray, Dict]:
        """Forward pass of the network.

        Args:
            input_data: input data of the shape (num_inputs). In case of a single scalar input it is
                directly cast to and interpreted like a one-element array.
            weights: trainable weights of the shape (num_weights). In case of a single scalar weight
                it is directly cast to and interpreted like a one-element array.
        Returns:
            The result of the neural network of the shape (output_shape).
        """
        input_, shape = self._validate_input(input_data)
        weights_ = self._validate_weights(weights)
        forward_data = self._forward(input_, weights_)
        return forward_data.reshape((*shape[:-1], *self._output_shape))

    @abstractmethod
    def _forward(self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
                 ) -> Union[np.ndarray, Dict]:
        raise NotImplementedError

    def backward(self, input_data: Optional[Union[List[float], np.ndarray, float]],
                 weights: Optional[Union[List[float], np.ndarray, float]]
                 ) -> Tuple[Optional[Union[np.ndarray, List[Dict]]],
                            Optional[Union[np.ndarray, List[Dict]]]]:
        """Backward pass of the network.

        Args:
            input_data: input data of the shape (num_inputs). In case of a
                single scalar input it is directly cast to and interpreted like a one-element array.
            weights: trainable weights of the shape (num_weights). In case of a single scalar weight
            it is directly cast to and interpreted like a one-element array.
        Returns:
            The result of the neural network of the backward pass, i.e., a tuple with the gradients
            for input and weights of shape (output_shape, num_input) and
            (output_shape, num_weights), respectively.
        """
        input_, shape = self._validate_input(input_data)
        weights_ = self._validate_weights(weights)
        # todo: reshape input gradients.
        return self._backward(input_, weights_)

    @abstractmethod
    def _backward(self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
                  ) -> Tuple[Optional[Union[np.ndarray, List[Dict]]],
                             Optional[Union[np.ndarray, List[Dict]]]]:
        raise NotImplementedError
