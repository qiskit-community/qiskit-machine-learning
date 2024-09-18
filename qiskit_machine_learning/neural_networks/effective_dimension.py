# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""An implementation of the effective dimension algorithm."""

import logging
import time
from typing import Union, List, Tuple

import numpy as np
from scipy.special import logsumexp

from ..utils import algorithm_globals
from ..exceptions import QiskitMachineLearningError
from .estimator_qnn import EstimatorQNN
from .neural_network import NeuralNetwork

logger = logging.getLogger(__name__)


class EffectiveDimension:
    """
    This class computes the global effective dimension for a Qiskit
    :class:`~qiskit_machine_learning.neural_networks.NeuralNetwork`
    following the definition used in [1].

        **References**
        [1]: Abbas et al., The power of quantum neural networks.
        `The power of QNNs <https://arxiv.org/pdf/2011.00027.pdf>`__.
    """

    def __init__(
        self,
        qnn: NeuralNetwork,
        weight_samples: Union[np.ndarray, int] = 1,
        input_samples: Union[np.ndarray, int] = 1,
    ) -> None:
        """
        Args:
            qnn: A Qiskit :class:`~qiskit_machine_learning.neural_networks.NeuralNetwork`,
                with a specific dimension ``(num_weights)`` that will determine the shape of the
                Fisher Information Matrix ``(num_input_samples * num_weight_samples, num_weights,
                num_weights)`` used to compute the global effective dimension for a set of
                ``input_samples``, of shape ``(num_input_samples, qnn_input_size)``, and
                ``weight_samples``, of shape ``(num_weight_samples, num_weights)``.
            weight_samples: An array of neural network parameters (weights), of shape
                ``(num_weight_samples, num_weights)``, or an ``int`` to indicate the number of
                parameter sets to sample randomly from a uniform distribution. By default,
                ``weight_samples = 1``.
            input_samples: An array of samples to the neural network, of shape
                ``(num_input_samples, qnn_input_size)``, or an ``int`` to indicate the number of
                input sets to sample randomly from a normal distribution. By default,
                ``input_samples = 1``.
        """

        # Store arguments
        self._weight_samples = None
        self._input_samples = None
        self._num_weight_samples = 1
        self._num_input_samples = 1
        self._model = qnn

        # Define weight samples and input samples
        self.weight_samples = weight_samples  # type: ignore
        # input setter uses self._model
        self.input_samples = input_samples  # type: ignore

    @property
    def weight_samples(self) -> np.ndarray:
        """Returns network weight samples."""
        return self._weight_samples

    @weight_samples.setter
    def weight_samples(self, weight_samples: Union[np.ndarray, int]) -> None:
        """Sets network weight samples."""
        if isinstance(weight_samples, int):
            # random sampling from uniform distribution
            self._weight_samples = algorithm_globals.random.uniform(
                0, 1, size=(weight_samples, self._model.num_weights)
            )
        else:
            # to be sure we have an array
            weight_samples = np.asarray(weight_samples)
            if len(weight_samples.shape) != 2 or weight_samples.shape[1] != self._model.num_weights:
                raise QiskitMachineLearningError(
                    f"The Effective Dimension class expects"
                    f" a weight_samples array of shape (M, qnn.num_weights)."
                    f" Got {weight_samples.shape}."
                )
            self._weight_samples = weight_samples

        self._num_weight_samples = len(self._weight_samples)

    @property
    def input_samples(self) -> np.ndarray:
        """Returns network input samples."""
        return self._input_samples

    @input_samples.setter
    def input_samples(self, input_samples: Union[np.ndarray, int]) -> None:
        """Sets network input samples."""
        if isinstance(input_samples, int):
            # random sampling from normal distribution
            self._input_samples = algorithm_globals.random.normal(
                0, 1, size=(input_samples, self._model.num_inputs)
            )
        else:
            # to be sure we have an array
            input_samples = np.asarray(input_samples)
            if len(input_samples.shape) != 2 or input_samples.shape[1] != self._model.num_inputs:
                raise QiskitMachineLearningError(
                    f"The Effective Dimension class expects"
                    f" an input sample array of shape (N, qnn.num_inputs)."
                    f" Got {input_samples.shape}."
                )
            self._input_samples = input_samples

        self._num_input_samples = len(self._input_samples)

    def run_monte_carlo(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method computes the model's Monte Carlo sampling for a set of input samples and
        weight samples.

        Returns:
             grads: QNN gradient vector, result of backward passes, of shape
                ``(num_input_samples * num_weight_samples, output_size, num_weights)``.
             outputs: QNN output vector, result of forward passes, of shape
                ``(num_input_samples * num_weight_samples, output_size)``.
        """
        grads = np.zeros(
            (
                self._num_input_samples * self._num_weight_samples,
                self._model.output_shape[0],
                self._model.num_weights,
            )
        )
        outputs = np.zeros(
            (self._num_input_samples * self._num_weight_samples, self._model.output_shape[0])
        )

        for i, param_set in enumerate(self._weight_samples):
            t_before_forward = time.time()
            forward_pass = np.asarray(
                self._model.forward(input_data=self._input_samples, weights=param_set)
            )
            t_after_forward = time.time()

            backward_pass = np.asarray(
                self._model.backward(input_data=self._input_samples, weights=param_set)[1]
            )
            t_after_backward = time.time()

            t_forward = t_after_forward - t_before_forward
            t_backward = t_after_backward - t_after_forward
            logger.debug(
                "Weight sample: %d, forward time: %.3f (s), backward time: %.3f (s)",
                i,
                t_forward,
                t_backward,
            )

            grads[self._num_input_samples * i : self._num_input_samples * (i + 1)] = backward_pass
            outputs[self._num_input_samples * i : self._num_input_samples * (i + 1)] = forward_pass

        # post-processing in the case of EstimatorQNN output, to match
        # the SamplerQNN output format
        if isinstance(self._model, EstimatorQNN):
            grads = np.concatenate([grads / 2, -1 * grads / 2], 1)
            outputs = np.concatenate([(outputs + 1) / 2, (1 - outputs) / 2], 1)

        return grads, outputs

    def get_fisher_information(
        self, gradients: np.ndarray, model_outputs: np.ndarray
    ) -> np.ndarray:
        """
        This method computes the average Jacobian for every set of gradients and model output as
        shown in Abbas et al.

        Args:
            gradients: A numpy array, result of the neural network's backward pass, of
                shape ``(num_input_samples * num_weight_samples, output_size, num_weights)``.
            model_outputs: A numpy array, result of the neural networks' forward pass,
                of shape ``(num_input_samples * num_weight_samples, output_size)``.
        Returns:
            fisher: A numpy array of shape
                ``(num_input_samples * num_weight_samples, num_weights, num_weights)``
                with the average Jacobian  for every set of gradients and model output given.
        """

        if model_outputs.shape < gradients.shape:
            # add dimension to model outputs for broadcasting
            model_outputs = np.expand_dims(model_outputs, axis=2)

        # get grad-vectors (gradient_k/model_output_k)
        # multiply by sqrt(model_output) so that the outer product cross term is correct
        # after Einstein summation
        gradvectors = np.sqrt(model_outputs) * gradients / model_outputs

        # compute the sum of matrices obtained from outer product of grad-vectors
        fisher_information = np.einsum("ijk,lji->ikl", gradvectors, gradvectors.T)

        return fisher_information

    def get_normalized_fisher(self, normalized_fisher: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        This method computes the normalized Fisher Information Matrix and extracts its trace.

        Args:
            normalized_fisher: The Fisher Information Matrix to be normalized.

        Returns:
             normalized_fisher: The normalized Fisher Information Matrix, a numpy array of size
                ``(num_input_samples, num_weights, num_weights)``.
             fisher_trace: The trace of the Fisher Information Matrix
                (before normalizing).
        """

        # compute the trace with all normalized_fisher
        fisher_trace = np.trace(np.average(normalized_fisher, axis=0))

        # average the normalized_fisher over the num_input_samples to get
        # the empirical normalized_fisher
        fisher_avg = np.average(
            np.reshape(
                normalized_fisher,
                (
                    self._num_weight_samples,
                    self._num_input_samples,
                    self._model.num_weights,
                    self._model.num_weights,
                ),
            ),
            axis=1,
        )

        # calculate normalized_normalized_fisher for all the empirical normalized_fisher
        normalized_fisher = self._model.num_weights * fisher_avg / fisher_trace
        return normalized_fisher, fisher_trace

    def _get_effective_dimension(
        self,
        normalized_fisher: np.ndarray,
        dataset_size: Union[List[int], np.ndarray, int],
    ) -> Union[np.ndarray, int]:
        if not isinstance(dataset_size, int) and len(dataset_size) > 1:
            # expand dims for broadcasting
            normalized_fisher = np.expand_dims(normalized_fisher, axis=0)
            n_expanded = np.expand_dims(np.asarray(dataset_size), axis=(1, 2, 3))
            logsum_axis = 1
        else:
            n_expanded = np.asarray(dataset_size)
            logsum_axis = None

        # calculate effective dimension for each data sample size out
        # of normalized normalized_fisher
        f_mod = normalized_fisher * n_expanded / (2 * np.pi * np.log(n_expanded))
        one_plus_fmod = np.eye(self._model.num_weights) + f_mod
        # take log. of the determinant because of overflow
        dets = np.linalg.slogdet(one_plus_fmod)[1]
        # divide by 2 because of square root
        dets_div = dets / 2
        effective_dims = (
            2
            * (logsumexp(dets_div, axis=logsum_axis) - np.log(self._num_weight_samples))
            / np.log(dataset_size / (2 * np.pi * np.log(dataset_size)))
        )

        return np.squeeze(effective_dims)

    def get_effective_dimension(
        self, dataset_size: Union[List[int], np.ndarray, int]
    ) -> Union[np.ndarray, int]:
        """
        This method computes the effective dimension for a dataset of size ``dataset_size``. If an
        array is passed, then effective dimension computed for each value in the array.

        Args:
            dataset_size: array of data sizes or a single integer value.

        Returns:
             effective_dim: array of effective dimensions for each dataset size in ``num_data``.
        """

        # step 1: Monte Carlo sampling
        grads, output = self.run_monte_carlo()

        # step 2: compute as many fisher info. matrices as (input, params) sets
        fisher = self.get_fisher_information(gradients=grads, model_outputs=output)

        # step 3: get normalized fisher info matrices
        normalized_fisher, _ = self.get_normalized_fisher(fisher)

        # step 4: compute eff. dim
        effective_dimensions = self._get_effective_dimension(normalized_fisher, dataset_size)

        return effective_dimensions


class LocalEffectiveDimension(EffectiveDimension):
    """
    This class computes the local effective dimension for a Qiskit
    :class:`~qiskit_machine_learning.neural_networks.NeuralNetwork`
    following the definition used in [1].

    In the local version of the algorithm the number of weight samples is limited to 1. Thus,
    ``weight_samples`` must be of the shape ``(1, qnn.num_weights)``.

        **References**
        [1]: Abbas et al., The power of quantum neural networks.
        `The power of QNNs <https://arxiv.org/pdf/2011.00027.pdf>`__.
    """

    # override setter to enforce 1 set of parameters
    @property
    def weight_samples(self) -> np.ndarray:
        """Returns network parameters."""
        return self._weight_samples

    @weight_samples.setter
    def weight_samples(self, weight_samples: Union[np.ndarray, int]) -> None:
        """Sets network parameters."""
        if isinstance(weight_samples, int):
            # random sampling from uniform distribution
            self._weight_samples = algorithm_globals.random.uniform(
                0, 1, size=(1, self._model.num_weights)
            )
        else:
            # there is a weird mypy error if we keep the same variable name, so there's 'weights'
            weights = np.asarray(weight_samples)
            # additional check to accept 1D arrays
            if len(weights.shape) < 2:
                weights = np.expand_dims(weight_samples, 0)
            if weights.shape[0] != 1 or weights.shape[1] != self._model.num_weights:
                raise QiskitMachineLearningError(
                    f"The Local Effective Dimension class expects"
                    f" a weight_samples array of shape (1, qnn.num_weights) or (qnn.num_weights)."
                    f" Got {weights.shape}."
                )
            self._weight_samples = weights

        self._num_weight_samples = 1
