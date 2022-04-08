# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""An implementation of the effective dimension algorithm."""

import time
from typing import Optional, Union, List, Callable, Tuple

import numpy as np
from scipy.special import logsumexp

from ..neural_networks import OpflowQNN, NeuralNetwork
from ..exceptions import QiskitMachineLearningError

from qiskit.utils import algorithm_globals

class EffectiveDimension:

    """
    This class computes the global effective dimension for Qiskit
    :class:`~qiskit_machine_learning.neural_networks.NeuralNetwork`s following the algorithm
    presented in "The Power of Quantum Neural Networks": https://arxiv.org/abs/2011.00027.
    """

    def __init__(
        self,
        qnn: NeuralNetwork,
        params: Optional[Union[List[float], np.ndarray, float]] = None,
        inputs: Optional[Union[List[float], np.ndarray, float]] = None,
        num_params: int = 1,
        num_inputs: int = 1,
        seed: int = 0,
        callback: Optional[Callable[[int, float, float], None]] = None
    ) -> None:

        """
        Args:
            qnn: A Qiskit :class:`~qiskit_machine_learning.neural_networks.NeuralNetwork`,
                with a specific dimension (num_weights) that will determine the shape of the
                Fisher Information Matrix (num_inputs * num_params, num_weights, num_weights)
                used to compute the global effective dimension for a set of ``inputs``,
                of shape (num_inputs, qnn_input_size),
                and ``params``, of shape (num_params, num_weights).
            params: An array of neural network parameters (weights), of shape
                (num_params, num_weights).
            inputs: An array of inputs to the neural network, of shape
                (num_inputs, qnn_input_size).
            num_params: If ``params`` is not provided, the algorithm will
                randomly sample ``num_params`` parameter sets from a
                uniform distribution. By default, num_params = 1.
            num_inputs:  If ``inputs`` is not provided, the algorithm will
                randomly sample ``num_inputs`` input sets
                from a normal distribution. By default, num_inputs = 1.
            callback: A callback function for the Monte Carlo sampling.
        """

        # Store arguments
        self._params = None
        self._inputs = None
        self._seed = seed
        self._num_params = num_params
        self._num_inputs = num_inputs
        self._model = qnn
        self._callback = callback

        # Define inputs and parameters
        self.params = params  # type: ignore
        # input setter uses self._model
        self.inputs = inputs  # type: ignore

        np.random.seed(0)

    def num_weights(self) -> int:
        """Returns the dimension of the model according to the definition
        from the original paper."""
        return self._model.num_weights

    @property
    def seed(self) -> int:
        """
        Get seed.
        """
        return self._seed

    @seed.setter
    def seed(self, seed: int) -> None:
        """
        Set seed.

        Args:
            seed (int): seed to use.
        """
        self._seed = seed

    @property
    def params(self) -> np.ndarray:
        """Returns network parameters."""
        return self._params

    @params.setter
    def params(self, params: Optional[Union[List[float], np.ndarray, float]]) -> None:
        """Sets network parameters."""
        if params is not None:
            self._params = np.asarray(params)
            self._num_params = len(self._params)
        else:
            # random sampling from uniform distribution
            np.random.seed(self._seed)
            self._params = np.random.uniform(0, 1, size=(self._num_params, self._model.num_weights))

    @property
    def inputs(self) -> np.ndarray:
        """Returns network inputs."""
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: Optional[Union[List[float], np.ndarray, float]]) -> None:
        """Sets network inputs."""
        if inputs is not None:
            self._inputs = np.asarray(inputs)
            self._num_inputs = len(self._inputs)
        else:
            # random sampling from normal distribution
            np.random.seed(self._seed)
            self._inputs = np.random.normal(0, 1, size=(self._num_inputs, self._model.num_inputs))

    def run_montecarlo(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method computes the model's Monte Carlo sampling for a set of
        inputs and parameters (params).

        Returns:
             grads: QNN gradient vector, result of backward passes, of shape
                (num_inputs * num_params, output_size, num_weights)
             outputs: QNN output vector, result of forward passes, of shape
                (num_inputs * num_params, output_size)
        """
        grads = np.zeros(
            (
                self._num_inputs * self._num_params,
                self._model.output_shape[0],
                self._model.num_weights,
            )
        )
        outputs = np.zeros((self._num_inputs * self._num_params, self._model.output_shape[0]))

        for (i, param_set) in enumerate(self.params):
            t_before_forward = time.time()
            fwd_pass = np.asarray(self._model.forward(input_data=self.inputs, weights=param_set))
            t_after_forward = time.time()

            if self._callback is not None:
                msg = f'iteration {i}, time forward pass: {t_after_forward - t_before_forward}'
                self._callback(msg)

            back_pass = np.asarray(
                self._model.backward(input_data=self.inputs, weights=param_set)[1]
            )
            t_after_backward = time.time()

            if self._callback is not None:
                msg = f'iteration {i}, time backward pass: {t_after_backward - t_after_forward}'
                self._callback(msg)

            grads[self._num_inputs * i: self._num_inputs * (i + 1)] = back_pass
            outputs[self._num_inputs * i: self._num_inputs * (i + 1)] = fwd_pass

        # post-processing in the case of OpflowQNN output, to match the CircuitQNN output format
        if isinstance(self._model, OpflowQNN):
            grads = np.concatenate([grads / 2, -1 * grads / 2], 1)
            outputs = np.concatenate([(outputs + 1) / 2, (1 - outputs) / 2], 1)

        return grads, outputs

    def get_fisher(self, gradients: np.ndarray, model_outputs: np.ndarray) -> np.ndarray:

        """
        This method computes the average Jacobian for every set of gradients and
        model output as shown in Abbas et al.

        Args:
            gradients: A numpy array, result of the neural network's backward pass, of
                shape (num_inputs * num_params, output_size, num_weights)
            model_outputs: A numpy array, result of the neural networks' forward pass,
                of shape (num_inputs * num_params, output_size)
        Returns:
            normalized_fisher: A numpy array of shape (num_inputs * num_params, num_weights, num_weights)
                with the average Jacobian  for every set of gradients and model output given.
        """

        if model_outputs.shape < gradients.shape:
            # add dimension to model outputs for broadcasting
            model_outputs = np.expand_dims(model_outputs, axis=2)

        # get grad-vectors (gradient_k/model_output_k)
        # multiply by sqrt(model_output) so that the outer product cross term is correct
        # after Einstein summation
        gradvectors = np.sqrt(model_outputs) * gradients / model_outputs

        # compute sum of matrices obtained from outer product of grad-vectors
        fisher = np.einsum("ijk,lji->ikl", gradvectors, gradvectors.T)

        return fisher

    def get_normalized_fisher(self, normalized_fisher: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        This method computes the normalized Fisher Information Matrix (f_hat)
        and extracts its trace.
        Args:
            normalized_fisher: The Fisher Information Matrix to be normalized.
        Returns:
             normalized_fisher: The normalized Fisher Information Matrix, a numpy array
                    of size (num_inputs, num_weights, num_weights)
             fisher_trace: The trace of the Fisher Information Matrix
                            (before normalizing).
        """

        # compute the trace with all normalized_fisher
        fisher_trace = np.trace(np.average(normalized_fisher, axis=0))

        # average the normalized_fisher over the num_inputs to get the empirical normalized_fisher
        fisher_avg = np.average(
            np.reshape(
                normalized_fisher,
                (
                    self._num_params,
                    self._num_inputs,
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
        self, normalized_fisher: Union[List[float], np.ndarray], n: Union[List[int], np.ndarray, int]
    ) -> Union[np.ndarray, int]:

        if not isinstance(n, int) and len(n) > 1:
            # expand dims for broadcasting
            normalized_fisher = np.expand_dims(normalized_fisher, axis=0)
            n_expanded = np.expand_dims(np.asarray(n), axis=(1, 2, 3))
            logsum_axis = 1
        else:
            n_expanded = n
            logsum_axis = None

        # calculate effective dimension for each data sample size "n" out
        # of normalized normalized_fisher
        f_mod = normalized_fisher * n_expanded / (2 * np.pi * np.log(n_expanded))
        one_plus_fmod = np.eye(self._model.num_weights) + f_mod
        # take log. of the determinant because of overflow
        dets = np.linalg.slogdet(one_plus_fmod)[1]
        # divide by 2 because of square root
        dets_div = dets / 2
        effective_dims = (
            2
            * (logsumexp(dets_div, axis=logsum_axis) - np.log(self._num_params))
            / np.log(n / (2 * np.pi * np.log(n)))
        )

        return np.squeeze(effective_dims)

    def get_effective_dimension(self, n: Union[List[int], np.ndarray, int]) -> Union[np.ndarray, int]:
        """
        This method compute the effective dimension for a data sample size ``n``.

        Args:
            n: array of sample sizes
        Returns:
             effective_dim: array of effective dimensions for each sample size in n.
        """

        # step 1: Monte Carlo sampling
        grads, output = self.run_montecarlo()

        # step 2: compute as many fisher info. matrices as (input, params) sets
        normalized_fisher = self.get_fisher(gradients=grads, model_outputs=output)

        # step 3: get normalized fisher info matrices
        normalized_fisher, _ = self.get_normalized_fisher(normalized_fisher)

        # step 4: compute eff. dim
        effective_dimensions = self._get_effective_dimension(normalized_fisher, n)

        return effective_dimensions


class LocalEffectiveDimension(EffectiveDimension):
    """
    Computes the LOCAL effective dimension for a parametrized model.
    """

    def __init__(
        self,
        qnn: NeuralNetwork,
        params: Optional[Union[List[float], np.ndarray, float]] = None,
        inputs: Optional[Union[List[float], np.ndarray, float]] = None,
        num_inputs: int = 1,
        callback: Optional[Callable[[int, float, float], None]] = None,
    ) -> None:
        """
        Args:
            qnn: A Qiskit :class:`~qiskit_machine_learning.neural_networks.NeuralNetwork`,
                with a specific number of weights (qnn_num_weights) and input size
                (qnn_input_size).
            num_inputs:  Number of inputs, if user chooses to randomly sample
                from a normal distribution.
            params: An array of neural network weights, of shape (1, qnn_num_weights).
            inputs: An array of inputs to the neural network, of shape
                (num_inputs, qnn_input_size).

        Raises:
            QiskitMachineLearningError: If more than 1 set of parameters is inputted.
        """
        super().__init__(qnn, params, inputs, 1, num_inputs, callback)

    # override setter to enforce 1 set of parameters
    @property
    def params(self) -> np.ndarray:
        """Returns network parameters."""
        return self._params

    @params.setter
    def params(self, params: Optional[Union[List[float], np.ndarray, float]]) -> None:
        """Sets network parameters."""
        if params is not None:
            params = np.asarray(params)
            if len(params.shape) > 1 and params.shape[0] > 1:
                raise ValueError(
                    f'The local effective dimension algorithm uses only 1 set of parameters, '
                    f'got {params.shape[0]}'
                )
            else:
                self._params = params
                self._num_params = len(self._params)
        else:
            # random sampling from uniform distribution
            np.random.seed(self._seed)
            self._params = np.random.uniform(0, 1, size=(1, self._model.num_weights))