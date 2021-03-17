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
from typing import Tuple, Union, List, Optional, Dict, Any

import numpy as np

from .neural_network import NeuralNetwork


class SamplingNeuralNetwork(NeuralNetwork):
    """
    A Sampling Neural Network abstract class for all (quantum) neural networks within Qiskit's
    machine learning module that generate samples instead of (expected) values.
    """

    def __init__(self, num_inputs: int, num_weights: int, dense: bool,
                 output_shape: Union[int, Tuple[int, ...]],
                 return_samples: bool = False) -> None:
        """

        Args:
            num_inputs: The number of input features.
            num_weights: The number of trainable weights.
            dense: Whether the output is dense or sparse.
            output_shape: The shape of the output.
            return_samples: Determines whether the network returns a batch of samples or a sparse
                vector (dictionary) of probabilities in its forward pass. In case of probabilities,
                the backward pass returns the probability gradients, while it returns (None, None)
                in the case of samples.
        Raises:
            QiskitMachineLearningError: Invalid parameter values.
        """
        self._return_samples = return_samples
        super().__init__(num_inputs, num_weights, dense, output_shape)

    @property
    def return_samples(self) -> bool:
        """
        Returns:
             ``True`` if the network returns a batch of samples and ``False`` if a sparse
             vector (dictionary) of probabilities in its forward pass.
        """
        return self._return_samples

    def _forward(self, input_data: Optional[Union[List[float], np.ndarray, float]],
                 weights: Optional[Union[List[float], np.ndarray, float]]
                 ) -> Union[np.ndarray, Dict]:
        """Forward pass of the network. Returns an array of samples or the probabilities, depending
        on the setting. Format depends on the set interpret function.
        """
        if self._return_samples:
            return self.sample(input_data, weights)
        else:
            return self.probabilities(input_data, weights)

    def _backward(self, input_data: Optional[Union[List[float], np.ndarray, float]],
                  weights: Optional[Union[List[float], np.ndarray, float]]
                  ) -> Tuple[Optional[Union[np.ndarray, List[Dict]]],
                             Optional[Union[np.ndarray, List[Dict]]]]:
        """Backward pass of the network. Returns (None, None) in case of samples and the
        corresponding here probability gradients otherwise.
        """
        if self._return_samples:
            return None, None
        else:
            return self.probability_gradients(input_data, weights)

    def sample(self, input_data: Union[List[float], np.ndarray, float],
               weights: Union[List[float], np.ndarray, float]) -> np.ndarray:
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
        input_ = self._validate_input(input_data)
        weights_ = self._validate_weights(weights)
        return self._sample(input_, weights_)

    @abstractmethod
    def _sample(self, input_data: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Returns samples from the network."""
        raise NotImplementedError

    def probabilities(self, input_data: Union[List[float], np.ndarray, float],
                      weights: Union[List[float], np.ndarray, float]
                      ) -> Union[np.ndarray, Dict[str, float]]:
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
        input_ = self._validate_input(input_data)
        weights_ = self._validate_weights(weights)
        return self._probabilities(input_, weights_)

    @abstractmethod
    def _probabilities(self, input_data: np.ndarray, weights: np.ndarray
                       ) -> Union[np.ndarray, Dict[Any, float]]:
        """Returns the sample probabilities."""
        raise NotImplementedError

    def probability_gradients(self, input_data: Optional[Union[List[float], np.ndarray, float]],
                              weights: Optional[Union[List[float], np.ndarray, float]]
                              ) -> Tuple[Union[np.ndarray, List[Dict]],
                                         Union[np.ndarray, List[Dict]]]:
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
        input_ = self._validate_input(input_data)
        weights_ = self._validate_weights(weights)
        return self._probability_gradients(input_, weights_)

    @abstractmethod
    def _probability_gradients(self, input_data: np.ndarray, weights: np.ndarray
                               ) -> Tuple[Union[np.ndarray, List[Dict]],
                                          Union[np.ndarray, List[Dict]]]:
        """Returns the probability gradients."""
        raise NotImplementedError
