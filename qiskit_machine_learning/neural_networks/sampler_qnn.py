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

"""A Neural Network implementation based on the Sampler primitive."""

from __future__ import annotations
import logging

from numbers import Integral
from typing import Callable, cast, Iterable, Sequence

import numpy as np
from qiskit.algorithms.gradients import (
    BaseSamplerGradient,
    ParamShiftSamplerGradient,
    SamplerGradientResult,
)
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseSampler, SamplerResult, Sampler
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
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


class SamplerQNN(NeuralNetwork):
    """A neural network implementation based on the Sampler primitive.

    The ``SamplerQNN`` is a neural network that takes in a parametrized quantum circuit
    with designated parameters for input data and/or weights and translates the quasi-probabilities
    estimated by the :class:`~qiskit.primitives.Sampler` primitive into predicted classes. Quite
    often, a combined quantum circuit is used. Such a circuit is built from two circuits:
    a feature map, it provides input parameters for the network, and an ansatz (weight parameters).

    The output can be set up in different formats, and an optional post-processing step
    can be used to interpret the sampler's output in a particular context (e.g. mapping the
    resulting bitstring to match the number of classes).

    In this example the network maps the output of the quantum circuit to two classes via a custom
    `interpret` function:

    .. code-block::

        from qiskit import QuantumCircuit
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

        from qiskit_machine_learning.neural_networks import SamplerQNN

        num_qubits = 2
        feature_map = ZZFeatureMap(feature_dimension=num_qubits)
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=1)

        qc = QuantumCircuit(num_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)


        def parity(x):
            return "{:b}".format(x).count("1") % 2


        qnn = SamplerQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            interpret=parity,
            output_shape=2
        )

        qnn.forward(input_data=[1, 2], weights=[1, 2, 3, 4])

    The following attributes can be set via the constructor but can also be read and
    updated once the SamplerQNN object has been constructed.

    Attributes:

        sampler (BaseSampler): The sampler primitive used to compute the neural network's results.
        gradient (BaseSamplerGradient): A sampler gradient to be used for the backward pass.
    """

    def __init__(
        self,
        *,
        circuit: QuantumCircuit,
        sampler: BaseSampler | None = None,
        input_params: Sequence[Parameter] | None = None,
        weight_params: Sequence[Parameter] | None = None,
        sparse: bool = False,
        interpret: Callable[[int], int | tuple[int, ...]] | None = None,
        output_shape: int | tuple[int, ...] | None = None,
        gradient: BaseSamplerGradient | None = None,
        input_gradients: bool = False,
    ):
        """
        Args:
            sampler: The sampler primitive used to compute the neural network's results.
                If ``None`` is given, a default instance of the reference sampler defined
                by :class:`~qiskit.primitives.Sampler` will be used.
            circuit: The parametrized quantum circuit that generates the samples of this network.
            input_params: The parameters of the circuit corresponding to the input.
            weight_params: The parameters of the circuit corresponding to the trainable weights.
            sparse: Returns whether the output is sparse or not.
            interpret: A callable that maps the measured integer to another unsigned integer or
                tuple of unsigned integers. These are used as new indices for the (potentially
                sparse) output array. If no interpret function is
                passed, then an identity function will be used by this neural network.
            output_shape: The output shape of the custom interpretation. It is ignored if no custom
                interpret method is provided where the shape is taken to be
                ``2^circuit.num_qubits``..
            gradient: An optional sampler gradient to be used for the backward pass.
                If ``None`` is given, a default instance of
                :class:`~qiskit.algorithms.gradients.ParamShiftSamplerGradient` will be used.
            input_gradients: Determines whether to compute gradients with respect to input data.
                 Note that this parameter is ``False`` by default, and must be explicitly set to
                 ``True`` for a proper gradient computation when using
                 :class:`~qiskit_machine_learning.connectors.TorchConnector`.
        Raises:
            QiskitMachineLearningError: Invalid parameter values.
        """
        # set primitive, provide default
        if sampler is None:
            sampler = Sampler()
        self.sampler = sampler

        # set gradient
        if gradient is None:
            gradient = ParamShiftSamplerGradient(self.sampler)
        self.gradient = gradient

        self._circuit = circuit.copy()
        if len(self._circuit.clbits) == 0:
            self._circuit.measure_all()

        if input_params is None:
            input_params = []
        self._input_params = list(input_params)

        if weight_params is None:
            weight_params = []
        self._weight_params = list(weight_params)

        if sparse:
            _optionals.HAS_SPARSE.require_now("DOK")

        self.set_interpret(interpret, output_shape)
        self._input_gradients = input_gradients

        super().__init__(
            num_inputs=len(self._input_params),
            num_weights=len(self._weight_params),
            sparse=sparse,
            output_shape=self._output_shape,
            input_gradients=self._input_gradients,
        )

    @property
    def circuit(self) -> QuantumCircuit:
        """Returns the underlying quantum circuit."""
        return self._circuit

    @property
    def input_params(self) -> Sequence[Parameter]:
        """Returns the list of input parameters."""
        return self._input_params

    @property
    def weight_params(self) -> Sequence[Parameter]:
        """Returns the list of trainable weights parameters."""
        return self._weight_params

    @property
    def interpret(self) -> Callable[[int], int | tuple[int, ...]] | None:
        """Returns interpret function to be used by the neural network. If it is not set in
        the constructor or can not be implicitly derived, then ``None`` is returned."""
        return self._interpret

    def set_interpret(
        self,
        interpret: Callable[[int], int | tuple[int, ...]] | None = None,
        output_shape: int | tuple[int, ...] | None = None,
    ) -> None:
        """Change 'interpret' and corresponding 'output_shape'.

        Args:
            interpret: A callable that maps the measured integer to another unsigned integer or
                tuple of unsigned integers. See constructor for more details.
            output_shape: The output shape of the custom interpretation. It is ignored if no custom
                interpret method is provided where the shape is taken to be
                ``2^circuit.num_qubits``.
        """

        # derive target values to be used in computations
        self._output_shape = self._compute_output_shape(interpret, output_shape)
        self._interpret = interpret if interpret is not None else lambda x: x

    def _compute_output_shape(
        self,
        interpret: Callable[[int], int | tuple[int, ...]] | None = None,
        output_shape: int | tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        """Validate and compute the output shape."""

        # this definition is required by mypy
        output_shape_: tuple[int, ...] = (-1,)

        if interpret is not None:
            if output_shape is None:
                raise QiskitMachineLearningError(
                    "No output shape given; it's required when using custom interpret!"
                )
            if isinstance(output_shape, Integral):
                output_shape = int(output_shape)
                output_shape_ = (output_shape,)
            else:
                output_shape_ = output_shape  # type: ignore
        else:
            if output_shape is not None:
                # Warn user that output_shape parameter will be ignored
                logger.warning(
                    "No interpret function given, output_shape will be automatically "
                    "determined as 2^num_qubits."
                )
            output_shape_ = (2**self._circuit.num_qubits,)

        return output_shape_

    def _postprocess(self, num_samples: int, result: SamplerResult) -> np.ndarray | SparseArray:
        """
        Post-processing during forward pass of the network.
        """

        if self._sparse:
            # pylint: disable=import-error
            from sparse import DOK

            prob = DOK((num_samples, *self._output_shape))
        else:
            prob = np.zeros((num_samples, *self._output_shape))

        for i in range(num_samples):
            counts = result.quasi_dists[i]
            shots = sum(counts.values())

            # evaluate probabilities
            for b, v in counts.items():
                key = self._interpret(b)
                if isinstance(key, Integral):
                    key = (cast(int, key),)
                key = (i, *key)  # type: ignore
                prob[key] += v / shots

        if self._sparse:
            return prob.to_coo()
        else:
            return prob

    def _postprocess_gradient(
        self, num_samples: int, results: SamplerGradientResult
    ) -> tuple[np.ndarray | SparseArray | None, np.ndarray | SparseArray]:
        """
        Post-processing during backward pass of the network.
        """

        if self._sparse:
            # pylint: disable=import-error
            from sparse import DOK

            input_grad = (
                DOK((num_samples, *self._output_shape, self._num_inputs))
                if self._input_gradients
                else None
            )
            weights_grad = DOK((num_samples, *self._output_shape, self._num_weights))
        else:

            input_grad = (
                np.zeros((num_samples, *self._output_shape, self._num_inputs))
                if self._input_gradients
                else None
            )
            weights_grad = np.zeros((num_samples, *self._output_shape, self._num_weights))

        if self._input_gradients:
            num_grad_vars = self._num_inputs + self._num_weights
        else:
            num_grad_vars = self._num_weights

        for sample in range(num_samples):
            for i in range(num_grad_vars):
                grad = results.gradients[sample][i]
                for k, val in grad.items():
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
                            input_grad[key] += val
                        else:
                            weights_grad[key] += val
                    else:
                        weights_grad[key] += val

        if self._sparse:
            if self._input_gradients:
                input_grad = input_grad.to_coo()  # pylint: disable=no-member
            weights_grad = weights_grad.to_coo()

        return input_grad, weights_grad

    def _forward(
        self,
        input_data: np.ndarray | None,
        weights: np.ndarray | None,
    ) -> np.ndarray | SparseArray | None:
        """
        Forward pass of the network.
        """
        parameter_values, num_samples = self._preprocess_forward(input_data, weights)

        if num_samples is not None and np.prod(parameter_values.shape) > 0:
            # sampler allows batching
            job = self.sampler.run([self._circuit] * num_samples, parameter_values)
            try:
                results = job.result()
            except Exception as exc:
                raise QiskitMachineLearningError("Sampler job failed.") from exc
            result = self._postprocess(num_samples, results)
        else:
            result = None

        return result

    def _backward(
        self,
        input_data: np.ndarray | None,
        weights: np.ndarray | None,
    ) -> tuple[np.ndarray | SparseArray | None, np.ndarray | SparseArray | None]:

        """Backward pass of the network."""
        # prepare parameters in the required format
        parameter_values, num_samples = self._preprocess_forward(input_data, weights)

        results = None
        if num_samples is not None and np.prod(parameter_values.shape) > 0:
            if self._input_gradients:
                job = self.gradient.run([self._circuit] * num_samples, parameter_values)
                try:
                    results = job.result()
                except Exception as exc:
                    raise QiskitMachineLearningError("Sampler job failed.") from exc
            else:
                if len(parameter_values[0]) > self._num_inputs:
                    job = self.gradient.run(
                        [self._circuit] * num_samples,
                        parameter_values,
                        parameters=[self._circuit.parameters[self._num_inputs :]] * num_samples,
                    )
                    try:
                        results = job.result()
                    except Exception as exc:
                        raise QiskitMachineLearningError("Sampler job failed.") from exc

        if results is None:
            return None, None

        input_grad, weights_grad = self._postprocess_gradient(num_samples, results)
        return input_grad, weights_grad  # `None` for gradients wrt input data, see TorchConnector
