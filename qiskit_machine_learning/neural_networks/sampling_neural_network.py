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

"""A Sampling Neural Network abstract class."""

from abc import abstractmethod
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


from .neural_network import NeuralNetwork


class SamplingNeuralNetwork(NeuralNetwork):
    """
    A Sampling Neural Network abstract class for all (quantum) neural networks within Qiskit's
    machine learning module that generate samples instead of (expected) values.
    """

    def __init__(
        self,
        num_inputs: int,
        num_weights: int,
        sparse: bool,
        sampling: bool,
        output_shape: Union[int, Tuple[int, ...]],
        input_gradients: bool = False,
    ) -> None:
        """

        Args:
            num_inputs: The number of input features.
            num_weights: The number of trainable weights.
            sparse: Returns whether the output is sparse or not.
            sampling: Determines whether the network returns a batch of samples or (possibly
                sparse) array of probabilities in its forward pass. In case of probabilities,
                the backward pass returns the probability gradients, while it returns (None, None)
                in the case of samples.
            output_shape: The shape of the output.
            input_gradients: Determines whether to compute gradients with respect to input data.
                Note that this parameter is ``False`` by default, and must be explicitly set to
                ``True`` for a proper gradient computation when using ``TorchConnector``.
        Raises:
            QiskitMachineLearningError: Invalid parameter values.
        """
        self._sampling = sampling
        super().__init__(num_inputs, num_weights, sparse, output_shape, input_gradients)

    @property
    def sampling(self) -> bool:
        """
        Returns:
             ``True`` if the network returns a batch of samples and ``False`` if a sparse
             vector (dictionary) of probabilities in its forward pass.
        """
        return self._sampling

    def _forward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Union[np.ndarray, SparseArray]:
        """Forward pass of the network. Returns an array of samples or the probabilities, depending
        on the setting. Format depends on the set interpret function.
        """
        if self._sampling:
            return self._sample(input_data, weights)
        else:
            return self._probabilities(input_data, weights)

    def _backward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Tuple[Optional[Union[np.ndarray, SparseArray]], Optional[Union[np.ndarray, SparseArray]],]:
        """Backward pass of the network. Returns (None, None) in case of samples and the
        corresponding here probability gradients otherwise.
        """
        if self._sampling:
            return None, None
        else:
            return self._probability_gradients(input_data, weights)

    def sample(
        self,
        input_data: Union[List[float], np.ndarray, float],
        weights: Union[List[float], np.ndarray, float],
    ) -> np.ndarray:
        """Samples from the network. Returns an array of samples. Format depends on the set
        interpret function.

        Args:
            input_data: input data of the shape (num_inputs). In case of a single scalar input it is
                directly cast to and interpreted like a one-element array.
            weights: trainable weights of the shape (num_weights). In case of a single scalar weight
                it is directly cast to and interpreted like a one-element array.
        Returns:
            The sample results of the neural network of the shape (output_shape).
        """
        input_, shape = self._validate_input(input_data)
        weights_ = self._validate_weights(weights)
        output_data = self._sample(input_, weights_)
        return self._validate_forward_output(output_data, shape)

    @abstractmethod
    def _sample(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> np.ndarray:
        """Returns samples from the network."""
        raise NotImplementedError

    def probabilities(
        self,
        input_data: Union[List[float], np.ndarray, float],
        weights: Union[List[float], np.ndarray, float],
    ) -> Union[np.ndarray, SparseArray]:
        """Histogram (as dict) of the samples from the network. Returns an array of samples. Format
        depends on the set interpret function.

        Args:
            input_data: input data of the shape (num_inputs). In case of a single scalar input it is
                directly cast to and interpreted like a one-element array.
            weights: trainable weights of the shape (num_weights). In case of a single scalar weight
                it is directly cast to and interpreted like a one-element array.
        Returns:
            The sample histogram of the neural network.
        """
        input_, shape = self._validate_input(input_data)
        weights_ = self._validate_weights(weights)
        output_data = self._probabilities(input_, weights_)
        return self._validate_forward_output(output_data, shape)

    @abstractmethod
    def _probabilities(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Union[np.ndarray, SparseArray]:
        """Returns the sample probabilities."""
        raise NotImplementedError

    def probability_gradients(
        self,
        input_data: Optional[Union[List[float], np.ndarray, float]],
        weights: Optional[Union[List[float], np.ndarray, float]],
    ) -> Tuple[Union[np.ndarray, SparseArray], Union[np.ndarray, SparseArray]]:
        """Probability gradients of histogram resulting from the network. Format depends on the set
        interpret function. Shape is (input_grad, weights_grad), where each grad has one dict for
        each parameter and each dict contains as value the derivative of the probability of
        measuring the key.

        Args:
            input_data: input data of the shape (num_inputs). In case of a single scalar input it is
                directly cast to and interpreted like a one-element array.
            weights: trainable weights of the shape (num_weights). In case of a single scalar weight
                it is directly cast to and interpreted like a one-element array.
        Returns:
            The probability gradients.
        """
        input_, shape = self._validate_input(input_data)
        weights_ = self._validate_weights(weights)
        input_grad, weight_grad = self._probability_gradients(input_, weights_)
        input_grad_reshaped, weight_grad_reshaped = self._validate_backward_output(
            input_grad, weight_grad, shape
        )

        return input_grad_reshaped, weight_grad_reshaped

    @abstractmethod
    def _probability_gradients(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Tuple[Union[np.ndarray, SparseArray], Union[np.ndarray, SparseArray]]:
        """Returns the probability gradients."""
        raise NotImplementedError
