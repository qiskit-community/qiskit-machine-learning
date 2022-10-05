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

"""Estimator quantum neural network class"""
import logging
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from qiskit.algorithms.gradients import BaseEstimatorGradient
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator

import qiskit_machine_learning.optionals as _optionals

from .neural_network import NeuralNetwork

if _optionals.HAS_SPARSE:
    # pylint: disable=import-error
    from sparse import SparseArray
else:

    class SparseArray:  # type: ignore
        """Empty SparseArray class
        Replacement if sparse.SparseArray is not present.
        """

        pass


logger = logging.getLogger(__name__)


class EstimatorQNN(NeuralNetwork):
    """A Neural Network implementation based on the Estimator primitive."""

    def __init__(
        self,
        estimator: BaseEstimator,
        circuit: QuantumCircuit,
        observables: Sequence[Union[BaseOperator, PauliSumOp]],
        input_params: Optional[Sequence[Parameter]] = None,
        weight_params: Optional[Sequence[Parameter]] = None,
        gradient: Optional[BaseEstimatorGradient] = None,
        input_gradients: bool = False,
    ):
        """
        Args:
            estimator: The estimator used to compute neural network's results.
            circuit: The quantum circuit to represent the neural network.
            observables: The observables for outputs of the neural network.
            input_params: The parameters that correspond to the input of the network.
            weight_params: The parameters that correspond to the trainable weights.
            gradient: The estimator gradient to be used for the backward pass.
            input_gradients: Determines whether to compute gradients with respect to input data.
                Note that this parameter is ``False`` by default, and must be explicitly set to
                ``True`` for a proper gradient computation when using ``TorchConnector``.
        """
        self._estimator = estimator
        self._circuit = circuit
        self._observables = observables
        self._input_params = list(input_params) or []
        self._weight_params = list(weight_params) or []
        self._gradient = gradient
        self.input_gradients = input_gradients

        super().__init__(
            len(self._input_params),
            len(self._weight_params),
            sparse=False,
            output_shape=len(observables),
            input_gradients=input_gradients,
        )

    @property
    def observables(self):
        """Returns the underlying observables of this QNN."""
        return self._observables

    @property
    def input_gradients(self) -> bool:
        """Returns whether gradients with respect to input data are computed by this neural network
        in the ``backward`` method or not. By default such gradients are not computed."""
        return self._input_gradients

    @input_gradients.setter
    def input_gradients(self, input_gradients: bool) -> None:
        """Turn on/off computation of gradients with respect to input data."""
        self._input_gradients = input_gradients

    def _preprocess(self, input_data, weights):
        """Pre-processing during forward pass of the network."""
        if len(input_data.shape) == 1:
            input_data = np.expand_dims(input_data, 0)
        num_samples = input_data.shape[0]
        # quick fix for 0 inputs
        if num_samples == 0:
            num_samples = 1

        parameter_values = []
        for i in range(num_samples):
            param_values = [input_data[i, j] for j, input_param in enumerate(self._input_params)]
            param_values += [weights[j] for j, weight_param in enumerate(self._weight_params)]
            parameter_values.append(param_values)

        return parameter_values, num_samples

    def _forward_postprocess(self, num_samples, results):
        """Post-processing during forward pass of the network."""
        res = np.zeros((num_samples, *self._output_shape))
        for i in range(num_samples):
            for j in range(self.output_shape[0]):
                res[i, j] = results.values[i * self.output_shape[0] + j]
        return res

    def _forward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Union[np.ndarray, SparseArray]:
        """Forward pass of the neural network."""
        parameter_values_, num_samples = self._preprocess(input_data, weights)
        parameter_values = [
            param_values for param_values in parameter_values_ for _ in range(self.output_shape[0])
        ]
        job = self._estimator.run(
            [self._circuit] * num_samples * self.output_shape[0],
            self._observables * num_samples,
            parameter_values,
        )
        results = job.result()
        return self._forward_postprocess(num_samples, results)

    def _backward_postprocess(self, num_samples, results):
        """Post-processing during backward pass of the network."""
        input_grad = (
            np.zeros((num_samples, *self.output_shape, self._num_inputs))
            if self._input_gradients
            else None
        )
        weights_grad = np.zeros((num_samples, *self.output_shape, self._num_weights))

        if self._input_gradients:
            num_grad_vars = self._num_inputs + self._num_weights
        else:
            num_grad_vars = self._num_weights

        for i in range(num_samples):
            for j in range(self.output_shape[0]):
                for k in range(num_grad_vars):
                    if self._input_gradients:
                        if k < self._num_inputs:
                            input_grad[i, j, k] = results.gradients[i * self.output_shape[0] + j][k]
                        else:
                            weights_grad[i, j, k - self._num_inputs] = results.gradients[
                                i * self.output_shape[0] + j
                            ][k]
                    else:
                        weights_grad[i, j, k] = results.gradients[i * self.output_shape[0] + j][k]

        return input_grad, weights_grad

    def _backward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],]:
        """Backward pass of the network."""
        # prepare parameters in the required format
        parameter_values_, num_samples = self._preprocess(input_data, weights)
        parameter_values = [
            param_values for param_values in parameter_values_ for _ in range(self.output_shape[0])
        ]
        if self._input_gradients:
            job = self._gradient.run(
                [self._circuit] * num_samples * self.output_shape[0],
                self._observables * num_samples,
                parameter_values,
            )
        else:
            job = self._gradient.run(
                [self._circuit] * num_samples * self.output_shape[0],
                self._observables * num_samples,
                parameter_values,
                parameters=[self._circuit.parameters[self._num_inputs :]]
                * num_samples
                * self.output_shape[0],
            )

        results = job.result()
        input_grad, weights_grad = self._backward_postprocess(num_samples, results)
        return input_grad, weights_grad  # `None` for gradients wrt input data, see TorchConnector
