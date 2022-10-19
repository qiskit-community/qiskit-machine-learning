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

from __future__ import annotations

import logging
from typing import Sequence, Tuple

import numpy as np
from qiskit.algorithms.gradients import BaseEstimatorGradient
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_machine_learning.exceptions import QiskitMachineLearningError

from .neural_network import NeuralNetwork

logger = logging.getLogger(__name__)


class EstimatorQNN(NeuralNetwork):
    """A Neural Network implementation based on the Estimator primitive."""

    def __init__(
        self,
        estimator: BaseEstimator,
        circuit: QuantumCircuit,
        observables: Sequence[BaseOperator | PauliSumOp],
        input_params: Sequence[Parameter] | None = None,
        weight_params: Sequence[Parameter] | None = None,
        gradient: BaseEstimatorGradient | None = None,
        input_gradients: bool = False,
    ):
        """
        Args:
            estimator: The estimator used to compute neural network's results.
            circuit: The quantum circuit to represent the neural network.
            observables: The observables for outputs of the neural network.
            input_params: The parameters that correspond to the input data of the network.
                If None, the input data is not bound to any parameters.
            weight_params: The parameters that correspond to the trainable weights.
                If None, the weights are not bound to any parameters.
            gradient: The estimator gradient to be used for the backward pass.
                If None, the gradient is not computed.
            input_gradients: Determines whether to compute gradients with respect to input data.
                Note that this parameter is ``False`` by default, and must be explicitly set to
                ``True`` for a proper gradient computation when using ``TorchConnector``.
        """
        self._estimator = estimator
        self._circuit = circuit
        self._observables = observables
        self._input_params = list(input_params) if input_params is not None else []
        self._weight_params = list(weight_params) if weight_params is not None else []
        self._gradient = gradient
        self._input_gradients = input_gradients

        super().__init__(
            len(self._input_params),
            len(self._weight_params),
            sparse=False,
            output_shape=len(observables),
            input_gradients=input_gradients,
        )

    @property
    def observables(self) -> Sequence[BaseOperator | PauliSumOp]:
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
        """Pre-processing during the forward pass and backward pass of the network."""
        if input_data is not None:
            num_samples = input_data.shape[0]
            if weights is not None:
                weights = np.broadcast_to(weights, (num_samples, len(weights)))
                parameters = np.concatenate((input_data, weights), axis=1)
            else:
                parameters = input_data
        else:
            if weights is not None:
                num_samples = 1
                parameters = np.broadcast_to(weights, (num_samples, len(weights)))
            else:
                return None, None
        return parameters, num_samples

    def _forward_postprocess(self, num_samples, results):
        """Post-processing during forward pass of the network."""
        res = np.zeros((num_samples, *self._output_shape))
        for i in range(num_samples):
            for j in range(self.output_shape[0]):
                res[i, j] = results.values[i * self.output_shape[0] + j]
        return res

    def _forward(
        self, input_data: np.ndarray | None, weights: np.ndarray | None
    ) -> np.ndarray | None:
        """Forward pass of the neural network."""
        parameter_values_, num_samples = self._preprocess(input_data, weights)
        if num_samples is None:
            return None
        else:
            parameter_values = [
                param_values for param_values in parameter_values_ for _ in range(self.output_shape[0])
            ]
            job = self._estimator.run(
                [self._circuit] * num_samples * self.output_shape[0],
                self._observables * num_samples,
                parameter_values,
            )
            try:
                results = job.result()
            except Exception as exc:
                raise QiskitMachineLearningError("Estimator job failed.") from exc

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
        self, input_data: np.ndarray | None, weights: np.ndarray | None
    ) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """Backward pass of the network."""
        # if no gradient is set, return None
        if self._gradient is None:
            return None, None
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

        try:
            results = job.result()
        except Exception as exc:
            raise QiskitMachineLearningError("Estimator job failed.") from exc

        input_grad, weights_grad = self._backward_postprocess(num_samples, results)
        return input_grad, weights_grad  # `None` for gradients wrt input data, see TorchConnector
