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


class EffectiveDimension:

    """
    This class computes the global effective dimension for Qiskit NeuralNetworks following the algorithm
    presented in "The Power of Quantum Neural Networks": https://arxiv.org/abs/2011.00027.
    """

    def __init__(
        self,
        qnn: NeuralNetwork,
        params: Optional[Union[List[float], np.ndarray, float]] = None,
        inputs: Optional[Union[List[float], np.ndarray, float]] = None,
        num_params: int = 1,
        num_inputs: int = 1,
        fix_seed=False,
        callback: Optional[Callable] = None,
    ) -> None:
        # pylint: disable=wrong-spelling-in-docstring
        """
        Args:
            qnn: A Qiskit ``NeuralNetwork``, with a specific number
                of weights/parameters (d = qnn_num_weights) that
                will determine the shape of the Fisher Information Matrix
                (num_inputs * num_params, d, d) used to compute the global
                effective dimension for a set of ``inputs``, of shape
                (num_inputs, qnn_input_size), and ``params``, of shape
                (num_params, d).
            params: An array of neural network parameters (weights), of shape
                (num_params, qnn_num_weights).
            inputs: An array of inputs to the neural network, of shape
                (num_inputs, qnn_input_size).
            num_params: If ``params`` is not provided, the algorithm will
                randomly sample ``num_params`` parameter sets from a
                uniform distribution. By default, num_params = 1.
            num_inputs:  If ``inputs`` is not provided, the algorithm will
                randomly sample ``num_inputs`` input sets
                from a normal distribution. By default, num_inputs = 1.
            callback: A callback function for the Montecarlo sampling.
        """

        # Store arguments
        self._params = None
        self._inputs = None
        self._num_params = num_params
        self._num_inputs = num_inputs
        self._model = qnn
        self._callback = callback

        if fix_seed:
            self._seed = 0
        else:
            self._seed = None
        # Define inputs and parameters
        self.params = params  # type: ignore
        # input setter uses self._model
        self.inputs = inputs  # type: ignore

    # keep d = num_weights for the sake of consistency with the
    # nomenclature in the original code/paper
    def d(self) -> int:  # pylint: disable=invalid-name
        """Returns the dimension of the model according to the definition
        from the original paper."""
        return self._model.num_weights

    @property
    def num_params(self) -> int:
        """Returns the number of network parameter sets."""
        return self._num_params

    @num_params.setter
    def num_params(self, num_params) -> None:
        """Sets the number of network parameter sets."""
        self._num_params = num_params

    @property
    def params(self) -> np.ndarray:
        """Returns network parameters."""
        return self._params

    @params.setter
    def params(self, params: Union[List[float], np.ndarray, float]) -> None:
        """Sets network parameters."""
        if params is not None:
            self._params = np.asarray(params)
            self.num_params = len(self._params)
        else:
            if self._seed is not None:
                np.random.seed(self._seed)
            # random sampling from uniform distribution
            params = np.random.uniform(0, 1, size=(self.num_params, self._model.num_weights))
            self._params = params

    @property
    def num_inputs(self) -> int:
        """Returns the number of input sets."""
        return self._num_inputs

    @num_inputs.setter
    def num_inputs(self, num_inputs) -> None:
        """Sets the number of input sets."""
        self._num_inputs = num_inputs

    @property
    def inputs(self) -> np.ndarray:
        """Returns network inputs."""
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: Union[List[float], np.ndarray, float]) -> None:
        """Sets network inputs."""
        if inputs is not None:
            self._inputs = np.asarray(inputs)
            self.num_inputs = len(self._inputs)
        else:
            if self._seed is not None:
                np.random.seed(self._seed)
            # random sampling from normal distribution
            self._inputs = np.random.normal(0, 1, size=(self.num_inputs, self._model.num_inputs))

    def run_montecarlo(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method computes the model's Monte Carlo sampling for a set of
        inputs and parameters (params).

        Returns:
             grads: QNN gradient vector, result of backward passes, of shape
                (num_inputs * num_params, output-size, d)
             outputs: QNN output vector, result of forward passes, of shape
                (num_inputs * num_params, output-size)
        """
        grads = np.zeros(
            (
                self.num_inputs * self.num_params,
                self._model.output_shape[0],
                self._model.num_weights,
            )
        )
        outputs = np.zeros((self.num_inputs * self.num_params, self._model.output_shape[0]))

        # could this be batched further?
        for (i, param_set) in enumerate(self.params):
            t_0 = time.time()
            back_pass = np.asarray(
                self._model.backward(input_data=self.inputs, weights=param_set)[1]
            )
            t_1 = time.time()
            fwd_pass = np.asarray(self._model.forward(input_data=self.inputs, weights=param_set))
            t_2 = time.time()

            grads[self.num_inputs * i : self.num_inputs * (i + 1)] = back_pass
            outputs[self.num_inputs * i : self.num_inputs * (i + 1)] = fwd_pass

            if self._callback is not None:
                self._callback(i, t_1 - t_0, t_2 - t_1)

        # post-processing in the case of OpflowQNN output, to match the CircuitQNN output format
        if isinstance(self._model, OpflowQNN):
            grads = np.concatenate([grads / 2, -1 * grads / 2], 1)
            outputs = np.concatenate([(outputs + 1) / 2, (1 - outputs) / 2], 1)

        return grads, outputs

    def get_fishers(self, gradients: np.ndarray, model_outputs: np.ndarray) -> np.ndarray:
        # pylint: disable=wrong-spelling-in-docstring
        """
        This method computes the average Jacobian for every set of gradients and
        model output given as:

            1/K(sum_k(sum_i gradients_i/sum_i model_output_i)) for i in len(gradients) for label k

        Args:
            gradients: A numpy array, result of the neural network's backward pass
            model_outputs: A numpy array, result of the neural networks' forward pass
        Returns:
            fishers: A numpy array with the average Jacobian (of shape dxd) for every
            set of gradients and model output given
        """

        if model_outputs.shape < gradients.shape:
            # add dimension to model outputs for broadcasting
            model_outputs = np.expand_dims(model_outputs, axis=2)

        # get grad-vectors (gradient_k/model_output_k)
        # multiply by sqrt(model_output) so that the outer product cross term is correct
        # after Einstein summation
        gradvectors = np.sqrt(model_outputs) * gradients / model_outputs

        # compute sum of matrices obtained from outer product of grad-vectors
        fishers = np.einsum("ijk,lji->ikl", gradvectors, gradvectors.T)

        return fishers

    def get_fhat(self, fishers: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        This method computes the normalized Fisher Information Matrix (f_hat)
        and extracts its trace.
        Args:
            fishers: The Fisher Information Matrix to be normalized.
        Returns:
             f_hat: The normalized Fisher Information Matrix, a numpy array
                    of size (num_inputs, d, d)
             fisher_trace: The trace of the Fisher Information Matrix
                            (before normalizing).
        """

        # compute the trace with all fishers
        fisher_trace = np.trace(np.average(fishers, axis=0))

        # average the fishers over the num_inputs to get the empirical fishers
        fisher_avg = np.average(
            np.reshape(
                fishers,
                (
                    self.num_params,
                    self.num_inputs,
                    self._model.num_weights,
                    self._model.num_weights,
                ),
            ),
            axis=1,
        )

        # calculate f_hats for all the empirical fishers
        f_hat = self._model.num_weights * fisher_avg / fisher_trace
        return f_hat, fisher_trace

    def _get_eff_dim(
        self, f_hat: Union[List[float], np.ndarray], n: Union[List[int], np.ndarray, int]
    ) -> Union[np.ndarray, int]:

        if not isinstance(n, int) and len(n) > 1:
            # expand dims for broadcasting
            f_hat = np.expand_dims(f_hat, axis=0)
            n_expanded = np.expand_dims(np.asarray(n), axis=(1, 2, 3))
            logsum_axis = 1
        else:
            n_expanded = n
            logsum_axis = None

        # calculate effective dimension for each data sample size "n" out
        # of normalized fishers
        f_mod = f_hat * n_expanded / (2 * np.pi * np.log(n_expanded))
        one_plus_fmod = np.eye(self._model.num_weights) + f_mod
        # take log. of the determinant because of overflow
        dets = np.linalg.slogdet(one_plus_fmod)[1]
        # divide by 2 because of square root
        dets_div = dets / 2
        effective_dims = (
            2
            * (logsumexp(dets_div, axis=logsum_axis) - np.log(self.num_params))
            / np.log(n / (2 * np.pi * np.log(n)))
        )

        return np.squeeze(effective_dims)

    def get_eff_dim(self, n: Union[List[int], np.ndarray, int]) -> Union[np.ndarray, int]:
        """
        This method compute the effective dimension for a data sample size ``n``.

        Args:
            n: array of sample sizes
        Returns:
             effective_dim: array of effective dimensions for each sample size in n.
        """

        # pylint: disable=wrong-spelling-in-comment
        # step 1: Montecarlo sampling
        grads, output = self.run_montecarlo()

        # step 2: compute as many fisher info. matrices as (input, params) sets
        fishers = self.get_fishers(gradients=grads, model_outputs=output)

        # step 3: get normalized fisher info matrices
        f_hat, _ = self.get_fhat(fishers)

        # step 4: compute eff. dim
        effective_dims = self._get_eff_dim(f_hat, n)

        return effective_dims


class LocalEffectiveDimension(EffectiveDimension):
    """
    Computes the LOCAL effective dimension for a parametrized model.
    """

    def __init__(
        self,
        qnn: NeuralNetwork,
        num_inputs: int = 1,
        params: Optional[Union[List, np.ndarray]] = None,
        inputs: Optional[Union[List, np.ndarray]] = None,
        fix_seed: bool = False,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            qnn: A Qiskit NeuralNetwork, with a specific number of weights
                (qnn_num_weights) and input size (qnn_input_size)
            num_inputs:  Number of inputs, if user chooses to randomly sample
                from a normal distribution.
            params: An array of neural network weights, of shape (1, qnn_num_weights).
            inputs: An array of inputs to the neural network, of shape
                (num_inputs, qnn_input_size).

        Raises:
            QiskitMachineLearningError: If ``len(params) > 1``
        """
        params = np.asarray(params)
        if params is not None and len(params.shape) > 1 and params.shape[0] > 1:
            raise QiskitMachineLearningError(
                "The local effective dimension algorithm uses only 1 set of parameters."
            )

        super().__init__(qnn, params, inputs, 1, num_inputs, fix_seed, callback)
