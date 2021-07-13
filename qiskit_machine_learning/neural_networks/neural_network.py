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
from typing import Tuple, Union, List, Optional

import numpy as np

try:
    from sparse import SparseArray
except ImportError:

    class SparseArray:  # type: ignore
        """Empty SparseArray class
        Replacement if sparse.SparseArray is not present.
        """

        pass


from ..exceptions import QiskitMachineLearningError


class NeuralNetwork(ABC):
    """Abstract Neural Network class providing forward and backward pass and handling
    batched inputs. This is to be implemented by other (quantum) neural networks.
    """

    def __init__(
        self,
        num_inputs: int,
        num_weights: int,
        sparse: bool,
        output_shape: Union[int, Tuple[int, ...]],
        input_gradients: bool = False,
    ) -> None:
        """
        Args:
            num_inputs: The number of input features.
            num_weights: The number of trainable weights.
            sparse: Determines whether the output is a sparse array or not.
            output_shape: The shape of the output.
            input_gradients: Determines whether to compute gradients with respect to input data.
        Raises:
            QiskitMachineLearningError: Invalid parameter values.
        """
        if num_inputs < 0:
            raise QiskitMachineLearningError(f"Number of inputs cannot be negative: {num_inputs}!")
        self._num_inputs = num_inputs

        if num_weights < 0:
            raise QiskitMachineLearningError(
                f"Number of weights cannot be negative: {num_weights}!"
            )
        self._num_weights = num_weights

        self._sparse = sparse

        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        if not np.all([s > 0 for s in output_shape]):
            raise QiskitMachineLearningError(
                f"Invalid output shape, all components must be > 0, but got: {output_shape}."
            )
        self._output_shape = output_shape

        self._input_gradients = input_gradients

    @property
    def num_inputs(self) -> int:
        """Returns the number of input features."""
        return self._num_inputs

    @property
    def num_weights(self) -> int:
        """Returns the number of trainable weights."""
        return self._num_weights

    @property
    def sparse(self) -> bool:
        """Returns whether the output is sparse or not."""
        return self._sparse

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Returns the output shape."""
        return self._output_shape

    @property
    def input_gradients(self) -> bool:
        """Returns whether gradients with respect to input data are computed by this neural network
        in the ``backward`` method or not. By default such gradients are not computed."""
        return self._input_gradients

    @input_gradients.setter
    def input_gradients(self, input_gradients: bool) -> None:
        """Turn on/off computation of gradients with respect to input data."""
        self._input_gradients = input_gradients

    def _validate_input(
        self, input_data: Optional[Union[List[float], np.ndarray, float]]
    ) -> Tuple[Union[np.ndarray, None], Union[Tuple[int, ...], None]]:
        if input_data is None:
            return None, None
        input_ = np.array(input_data)
        shape = input_.shape
        if len(shape) == 0:
            # there's a single value in the input.
            input_ = input_.reshape((1, 1))
            return input_, shape

        if shape[-1] != self._num_inputs:
            raise QiskitMachineLearningError(
                f"Input data has incorrect shape, last dimension "
                f"is not equal to the number of inputs: "
                f"{self._num_inputs}, but got: {shape[-1]}."
            )

        if len(shape) == 1:
            # add an empty dimension for samples (batch dimension)
            input_ = input_.reshape((1, -1))
        elif len(shape) > 2:
            # flatten lower dimensions, keep num_inputs as a last dimension
            input_ = input_.reshape((np.product(input_.shape[:-1]), -1))

        return input_, shape

    def _validate_weights(
        self, weights: Optional[Union[List[float], np.ndarray, float]]
    ) -> Union[np.ndarray, None]:
        if weights is None:
            return None
        weights_ = np.array(weights)
        return weights_.reshape(self._num_weights)

    def _validate_forward_output(
        self, output_data: np.ndarray, original_shape: Tuple[int, ...]
    ) -> np.ndarray:
        if original_shape and len(original_shape) >= 2:
            output_data = output_data.reshape((*original_shape[:-1], *self._output_shape))

        return output_data

    def _validate_backward_output(
        self,
        input_grad: np.ndarray,
        weight_grad: np.ndarray,
        original_shape: Tuple[int, ...],
    ) -> Tuple[Union[np.ndarray, SparseArray], Union[np.ndarray, SparseArray]]:
        if input_grad is not None and original_shape and len(original_shape) >= 2:
            input_grad = input_grad.reshape(
                (*original_shape[:-1], *self._output_shape, self._num_inputs)
            )
        if weight_grad is not None and original_shape and len(original_shape) >= 2:
            weight_grad = weight_grad.reshape(
                (*original_shape[:-1], *self._output_shape, self._num_weights)
            )

        return input_grad, weight_grad

    def forward(
        self,
        input_data: Optional[Union[List[float], np.ndarray, float]],
        weights: Optional[Union[List[float], np.ndarray, float]],
    ) -> Union[np.ndarray, SparseArray]:
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
        output_data = self._forward(input_, weights_)
        return self._validate_forward_output(output_data, shape)

    @abstractmethod
    def _forward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Union[np.ndarray, SparseArray]:
        raise NotImplementedError

    def backward(
        self,
        input_data: Optional[Union[List[float], np.ndarray, float]],
        weights: Optional[Union[List[float], np.ndarray, float]],
    ) -> Tuple[Optional[Union[np.ndarray, SparseArray]], Optional[Union[np.ndarray, SparseArray]],]:
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
        input_grad, weight_grad = self._backward(input_, weights_)

        input_grad_reshaped, weight_grad_reshaped = self._validate_backward_output(
            input_grad, weight_grad, shape
        )

        return input_grad_reshaped, weight_grad_reshaped

    @abstractmethod
    def _backward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Tuple[Optional[Union[np.ndarray, SparseArray]], Optional[Union[np.ndarray, SparseArray]],]:
        raise NotImplementedError
