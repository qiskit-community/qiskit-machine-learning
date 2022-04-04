# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""An implementation of the effective dimension algorithm."""

import numpy as np
import time

from scipy.special import logsumexp
from typing import Optional, Union, List, Callable, Tuple

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
        num_params: Optional[int] = 1,
        num_inputs: Optional[int] = 1,
        fix_seed = False,
        callback: Optional[Callable] = None
    ) -> None:
        """
        Args:
            qnn: A Qiskit ``NeuralNetwork``, with a specific number of weights/parameters (d = qnn_num_weights) that
                will determine the shape of the Fisher Information Matrix (num_inputs * num_params, d, d) used to
                compute the global effective dimension for a set of ``inputs``, of shape (num_inputs, qnn_input_size),
                and ``params``, of shape (num_params, d).
            params: An array of neural network parameters (weights), of shape (num_params, qnn_num_weights).
            inputs: An array of inputs to the neural network, of shape (num_inputs, qnn_input_size).
            num_params: If ``params`` is not provided, the algorithm will randomly sample ``num_params`` parameter sets
                from a uniform distribution. By default, num_params = 1.
            num_inputs:  If ``inputs`` is not provided, the algorithm will randomly sample ``num_inputs`` input sets
                from a normal distribution. By default, num_inputs = 1.
            callback: A callback function.
        """
        # Store arguments
        self._params = None
        self._inputs = None
        self._num_params = num_params
        self._num_inputs = num_inputs
        self._model = qnn
        self._callback = callback

        # Define parameter set size (d)
        self.d = self._model.num_weights

        if fix_seed:
            self._seed = 0
        else:
            self._seed = None
        # Define inputs and parameters
        self.params = params
        self.inputs = inputs # input setter uses self._model

    @property
    def num_params(self) -> int:
        return self._num_params

    @num_params.setter
    def num_params(self, num_params) -> None:
        self._num_params = num_params

    @property
    def params(self) -> np.ndarray:
        return self._params

    @params.setter
    def params(self, params: Union[List[float], np.ndarray, float]) -> None:
        if params is not None:
            self._params = np.asarray(params)
            self.num_params = len(self._params)
        else:
            if self._seed is not None:
                np.random.seed(self._seed)
            # random sampling from uniform distribution
            params = np.random.uniform(0, 1, size=(self.num_params, self.d))
            self._params = params

    @property
    def num_inputs(self) -> int:
        return self._num_inputs

    @num_inputs.setter
    def num_inputs(self, num_inputs) -> None:
        self._num_inputs = num_inputs

    @property
    def inputs(self) -> np.ndarray:
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: Union[List[float], np.ndarray, float]) -> None:
        if inputs is not None:
            self._inputs = np.asarray(inputs)
            self.num_inputs = len(self._inputs)
        else:
            if self._seed is not None:
                np.random.seed(self._seed)
            # random sampling from normal distribution
            self._inputs = np.random.normal(0, 1, size=(self.num_inputs, self._model.num_inputs))

    def run_montecarlo(self) -> [np.ndarray, np.ndarray]:
        """
        This method computes the model's Monte Carlo sampling for a set of inputs and parameters (params).

        Returns:
             grads: QNN gradient vector, result of backward passes, of shape (num_inputs * num_params, outputsize, d)
             outputs: QNN output vector, result of forward passes, of shape (num_inputs * num_params, outputsize)
        """
        grads = np.zeros((self.num_inputs * self.num_params, self._model.output_shape[0], self.d))
        outputs = np.zeros((self.num_inputs * self.num_params, self._model.output_shape[0]))

        # could this be batched further?
        for (i, p) in enumerate(self.params):
            t0 = time.time()
            back_pass = np.asarray(self._model.backward(input_data=self.inputs, weights=p)[1])
            t1 = time.time()
            fwd_pass = np.asarray(self._model.forward(input_data=self.inputs, weights=p))
            t2 = time.time()

            grads[self.num_inputs * i:self.num_inputs * (i + 1)] = back_pass
            outputs[self.num_inputs * i:self.num_inputs * (i + 1)] = fwd_pass

            if self._callback is not None:
                self._callback(i, t1-t0, t2-t1)

        # post-processing in the case of OpflowQNN output, to match the CircuitQNN output format
        if isinstance(self._model, OpflowQNN):
            grads = np.concatenate([grads/ 2, -1 * grads / 2], 1)
            outputs = np.concatenate([(outputs + 1) / 2, (1 - outputs) / 2], 1)

        return grads, outputs

    def get_fishers(
        self,
        gradients: np.ndarray,
        model_outputs: np.ndarray
    ) -> np.ndarray:
        """
        This method computes the average jacobian for every set of gradients and model output given as:

            1/K(sum_k(sum_i gradients_i/sum_i model_output_i)) for i in len(gradients) for label k

        Args:
            gradients: A numpy array, result of the neural network's backward pass
            model_outputs: A numpy array, result of the neural networks's forward pass
        Returns:
            fishers: A numpy array with the average jacobian (of shape dxd) for every set of gradients and model output given
        """

        if model_outputs.shape < gradients.shape:
            # add dimension to model outputs for broadcasting
            model_outputs = np.expand_dims(model_outputs, axis=2)

        # get gradvectors (gradient_k/model_output_k)
        # multiply by sqrt(model_output) so that the outer product cross term is correct after einsum
        gradvectors = np.sqrt(model_outputs) * gradients / model_outputs

        # compute sum of matrices obtained from outer product of gradvectors
        fishers = np.einsum('ijk,lji->ikl', gradvectors, gradvectors.T)

        return fishers

    def get_fhat(
            self,
            fishers: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        This method computes the normalized Fisher Information Matrix (f_hat) out of a and extracts its trace.
        Args:
            fishers: The FIM to be normalized
        Returns:
             f_hat: The normalized FIM, a numpy array of size (num_inputs, d, d)
             fisher_trace: The trace of the FIM (before normalizing)
        """

        # compute the trace with all fishers
        fisher_trace = np.trace(np.average(fishers, axis=0))

        # average the fishers over the num_inputs to get the empirical fishers
        fisher_avg = np.average(np.reshape(fishers, (self.num_params, self.num_inputs, self.d, self.d)), axis=1)

        # calculate f_hats for all the empirical fishers
        f_hat = self.d * fisher_avg / fisher_trace
        return f_hat, fisher_trace

    def _get_eff_dim(
            self,
            f_hat: Union[List[float], np.ndarray],
            n: Union[List[int], np.ndarray, int]
    ) -> Union[np.ndarray, int]:

        if type(n) is not int and len(n) > 1:
            # expand dims for broadcasting
            f_hat = np.expand_dims(f_hat, axis=0)
            ns = np.expand_dims(np.asarray(n), axis=(1,2,3))
            logsum_axis = 1
        else:
            ns = n
            logsum_axis = None

        # calculate effective dimension for each data sample size "n" out
        # of normalized fishers
        fs = f_hat * ns / (2 * np.pi * np.log(ns))
        one_plus_fs = np.eye(self.d) + fs
        dets = np.linalg.slogdet(one_plus_fs)[1]  # log det because of overflow
        rs = dets / 2  # divide by 2 because of sqrt
        effective_dims = 2 * (logsumexp(rs, axis=logsum_axis) - np.log(self.num_params)) / \
                         np.log(n / (2 * np.pi * np.log(n)))

        return np.squeeze(effective_dims)

    def get_eff_dim(
        self,
        n: Union[List[int], np.ndarray, int]
    ) -> Union[np.ndarray, int]:
        """
        This method compute the effective dimension for a data sample size ``n``.

        Args:
            n: array of sample sizes
        Returns:
             effective_dim: array of effective dimensions for each sample size in n
        """

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
    def __init__(self,
            qnn: NeuralNetwork,
            num_inputs: Optional[int] = 1,
            params: Optional[Union[List, np.ndarray]] = None,
            inputs: Optional[Union[List, np.ndarray]] = None,
            fix_seed: bool = False,
            callback: Optional[Callable] = None,
            ) -> None:
        """
        Args:
            qnn: A Qiskit NeuralNetwork, with a specific number of weights (qnn_num_weights) and input size (qnn_input_size)
            num_inputs:  Number of inputs, if user chooses to randomly sample from a normal distribution.
            params: An array of neural network weights, of shape (1, qnn_num_weights).
            inputs: An array of inputs to the neural network, of shape (num_inputs, qnn_input_size).

        Raises:
            ValueError: If len(params) > 1
        """
        if params is not None and len(params.shape) > 1 and params.shape[0] > 1:
                    raise QiskitMachineLearningError("The local effective dimension algorithm uses only 1 set of parameters.")

        super().__init__(qnn, num_inputs, params, inputs, fix_seed, callback)


