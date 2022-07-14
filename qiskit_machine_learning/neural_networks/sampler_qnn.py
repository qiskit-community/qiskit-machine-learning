# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A Neural Network implementation based on the Sampler primitive."""

from __future__ import annotations
import logging
from numbers import Integral
from typing import Optional, Union, List, Tuple, Callable, cast, Iterable

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.exceptions import QiskitMachineLearningError, QiskitError

logger = logging.getLogger(__name__)


class SamplerQNN:
    """A Neural Network implementation based on the Sampler primitive."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        input_params: List[Parameter] | None = None,
        weight_params: List[Parameter] | None = None,
        interpret: Callable[[int], int | Tuple[int, ...]] | None = None,
        output_shape: int | Tuple[int, ...] | None = None,
        sampler_factory: Callable | None = None,
        gradient_factory: Callable | str | None = None,
    )-> None:
        """

        Args:
            circuit: The parametrized quantum circuit that generates the samples of this network.
            input_params: The parameters of the circuit corresponding to the input.
            weight_params: The parameters of the circuit corresponding to the trainable weights.
            interpret: A callable that maps the measured integer to another unsigned integer or
                tuple of unsigned integers. These are used as new indices for the (potentially
                sparse) output array. If no interpret function is
                passed, then an identity function will be used by this neural network.
            output_shape: The output shape of the custom interpretation
            sampler_factory: Factory for sampler primitive
            gradient_factory: String indicating pre-implemented gradient method or factory for
                gradient class
            input_gradients: to be added
        Raises:
            QiskitMachineLearningError: Invalid parameter values.
        """

        self._circuit = circuit

        self._gradient_factory = gradient_factory
        self._sampler_factory = sampler_factory

        self._sampler = None
        self._gradient = None

        self._input_params = list(input_params or [])
        self._weight_params = list(weight_params or [])

        self.output_shape = None
        self._num_inputs = len(self._input_params)
        self._num_weights = len(self._weight_params)

        self.num_weights = self._num_weights

        # the circuit must always have measurements...
        # add measurements in case none are given
        if len(self._circuit.clbits) == 0:
            self._circuit.measure_all()

        self._interpret = interpret
        self._original_interpret = interpret

        # set interpret and compute output shape
        self.set_interpret(interpret, output_shape)

        # not implemented yet
        # self._input_gradients = None

    def __enter__(self):
        """
        QNN used with context managers.
        """
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        """
        QNN used with context managers.
        """
        self.close()

    def open(self) -> None:
        """
        Open sampler/gradient session.
        """
        # we should delay instantiation of the primitives until they are really required
        self._sampler = self._sampler_factory(
            circuits=self._circuit, parameters=[self._input_params + self._weight_params]
        )

        if self._gradient_factory is not None:
            # if gradient method -> sampler with gradient functionality
            # self._gradient = ParamShiftSamplerGradient(
            #     circuit = self._circuit,
            #     sampler = self._sampler_factory
            # )
            pass  # waiting for gradients

    def close(self) -> None:
        """
        Close sampler/gradient session.
        """
        self._sampler.__exit__()

    def set_interpret(
        self,
        interpret: Callable[[int], int| Tuple[int, ...]] | None = None,
        output_shape: int | Tuple[int, ...] | None = None
    ) -> None:
        """Change 'interpret' and corresponding 'output_shape'.

        Args:
            interpret: A callable that maps the measured integer to another unsigned integer or
                tuple of unsigned integers. See constructor for more details.
            output_shape: The output shape of the custom interpretation, only used in the case
                where an interpret function is provided. See constructor
                for more details.
        """

        # save original values
        self._original_output_shape = output_shape
        self._original_interpret = interpret

        # derive target values to be used in computations
        self._output_shape = self._compute_output_shape(interpret, output_shape)
        self._interpret = interpret if interpret is not None else lambda x: x
        self.output_shape = self._output_shape

    def _compute_output_shape(
        self,
        interpret: Callable[[int], int | Tuple[int, ...]] | None = None,
        output_shape: int | Tuple[int, ...] | None = None
    ) -> Tuple[int, ...]:
        """Validate and compute the output shape."""

        # this definition is required by mypy
        output_shape_: Tuple[int, ...] = (-1,)

        if interpret is not None:
            if output_shape is None:
                raise QiskitMachineLearningError(
                    "No output shape given, but required in case of custom interpret!"
                )
            if isinstance(output_shape, Integral):
                output_shape = int(output_shape)
                output_shape_ = (output_shape,)
            else:
                output_shape_ = output_shape
        else:
            if output_shape is not None:
                # Warn user that output_shape parameter will be ignored
                logger.warning(
                    "No interpret function given, output_shape will be automatically "
                    "determined as 2^num_qubits."
                )

            output_shape_ = (2**self._circuit.num_qubits,)

        return output_shape_

    def forward(
        self,
        input_data: List[float] | np.ndarray | float | None,
        weights: List[float] | np.ndarray | float | None,
    ) -> np.ndarray:
        """
        Forward pass of the network. Returns the probabilities.
        Format depends on the set interpret function.
        """
        result = self._forward(input_data, weights)
        return result

    def _preprocess(
        self,
        input_data: List[float] | np.ndarray | float | None,
        weights: List[float] | np.ndarray | float | None,
    ) -> Tuple[List[float], int]:
        """
        Pre-processing during forward pass of the network.
        """
        if len(input_data.shape) == 1:
            input_data = np.expand_dims(input_data, 0)
        num_samples = input_data.shape[0]
        # quick fix for 0 inputs
        if num_samples == 0:
            num_samples = 1

        parameters = []
        for i in range(num_samples):
            param_values = [input_data[i, j] for j, input_param in enumerate(self._input_params)]
            param_values += [weights[j] for j, weight_param in enumerate(self._weight_params)]
            parameters.append(param_values)

        return parameters, num_samples

    def _postprocess(self, num_samples, result):
        """
        Post-processing during forward pass of the network.
        """
        prob = np.zeros((num_samples, *self._output_shape))

        for i in range(num_samples):
            counts = result.quasi_dists[i]
            print(counts)
            shots = sum(counts.values())

            # evaluate probabilities
            for b, v in counts.items():
                key = (i, int(self._interpret(b)))  # type: ignore
                prob[key] += v / shots

        return prob

    def _forward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Forward pass of the network.
        """
        parameter_values, num_samples = self._preprocess(input_data, weights)

        # sampler allows batching (gradient doesn't)
        results = self._sampler([0] * num_samples, parameter_values)

        result = self._postprocess(num_samples, results)

        return result

    def backward(
        self,
        input_data: Optional[Union[List[float], np.ndarray, float]],
        weights: Optional[Union[List[float], np.ndarray, float]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],]:
        """Backward pass of the network. Returns (None, None) when no gradient is
         provided and the corresponding here probability gradients otherwise.
        """
        if self._gradient:
            return self._backward(input_data, weights)
        else:
            return None, None

    def _preprocess_gradient(self, input_data, weights):
        """
        Pre-processing during backward pass of the network.
        """
        if len(input_data.shape) == 1:
            input_data = np.expand_dims(input_data, 0)

        num_samples = input_data.shape[0]
        # quick fix for 0 inputs
        if num_samples == 0:
            num_samples = 1

        parameters = []
        for i in range(num_samples):

            param_values = [input_data[i, j] for j, input_param in enumerate(self._input_params)]
            param_values += [weights[j] for j, weight_param in enumerate(self._weight_params)]
            parameters.append(param_values)

        return parameters, num_samples

    def _postprocess_gradient(self, num_samples, results):
        """
        Post-processing during backward pass of the network.
        """
        input_grad = np.zeros((num_samples, 1, self._num_inputs)) if self._input_gradients else None
        weights_grad = np.zeros((num_samples, *self._output_shape, self._num_weights))

        if self._input_gradients:
            num_grad_vars = self._num_inputs + self._num_weights
        else:
            num_grad_vars = self._num_weights

        for sample in range(num_samples):

            for i in range(num_grad_vars):
                grad = results[sample].quasi_dists[i + self._num_inputs]
                for k in grad.keys():
                    val = results[sample].quasi_dists[i + self._num_inputs][k]
                    # get index for input or weights gradients
                    if self._input_gradients:
                        grad_index = i if i < self._num_inputs else i - self._num_inputs
                    else:
                        grad_index = i
                    # interpret integer and construct key
                    key = self._interpret(k)
                    if isinstance(key, Integral):
                        key = (sample, int(key), grad_index)
                    else:
                        # if key is an array-type, cast to hashable tuple
                        key = tuple(cast(Iterable[int], key))
                        key = (sample, *key, grad_index)
                    # store value for inputs or weights gradients
                    if self._input_gradients:
                        # we compute input gradients first
                        if i < self._num_inputs:
                            input_grad[key] += np.real(val)
                        else:
                            weights_grad[key] += np.real(val)
                    else:
                        weights_grad[key] += np.real(val)

        return input_grad, weights_grad

    def _backward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],]:

        """Backward pass of the network.
        """
        # prepare parameters in the required format
        parameter_values, num_samples = self._preprocess_gradient(input_data, weights)

        results = []
        for sample in range(num_samples):
            if self._input_gradients:
                result = self._gradient(parameter_values[sample])
            else:
                result = self._gradient(
                    parameter_values[sample],
                    partial=self._sampler._circuit.parameters[self._num_inputs :],
                )

            results.append(result)
        input_grad, weights_grad = self._postprocess_gradient(num_samples, results)

        return None, weights_grad  # `None` for gradients wrt input data, see TorchConnector
