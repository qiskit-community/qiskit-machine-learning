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

"""A Neural Network implementation based on the Sampler primitive."""

from __future__ import annotations
import logging
from numbers import Integral
from typing import Callable, cast, Iterable, Sequence
import numpy as np

from qiskit.primitives import BaseSamplerV1
from qiskit.primitives.base import BaseSamplerV2

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseSampler, SamplerResult, Sampler
from qiskit.result import QuasiDistribution
from qiskit.transpiler.passmanager import BasePassManager

import qiskit_machine_learning.optionals as _optionals

from ..gradients import (
    BaseSamplerGradient,
    ParamShiftSamplerGradient,
    SamplerGradientResult,
)
from ..circuit.library import QNNCircuit
from ..exceptions import QiskitMachineLearningError
from ..utils.deprecation import issue_deprecation_msg
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
    In this case a :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` can be passed as
    circuit to simplify the composition of a feature map and ansatz.
    If a :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` is passed as circuit, the
    input and weight parameters do not have to be provided, because these two properties are taken
    from the :class:`~qiskit_machine_learning.circuit.library.QNNCircuit`.

    The output can be set up in different formats, and an optional post-processing step
    can be used to interpret the sampler's output in a particular context (e.g. mapping the
    resulting bitstring to match the number of classes).

    In this example the network maps the output of the quantum circuit to two classes via a custom
    `interpret` function:

    .. code-block::

        from qiskit import QuantumCircuit
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        from qiskit_machine_learning.circuit.library import QNNCircuit

        from qiskit_machine_learning.neural_networks import SamplerQNN

        num_qubits = 2

        def parity(x):
            return f"{bin(x)}".count("1") % 2

        # Using the QNNCircuit:
        # Create a parameterized 2 qubit circuit composed of the default ZZFeatureMap feature map
        # and RealAmplitudes ansatz.
        qnn_qc = QNNCircuit(num_qubits)

        qnn = SamplerQNN(
            circuit=qnn_qc,
            interpret=parity,
            output_shape=2
        )

        qnn.forward(input_data=[1, 2], weights=[1, 2, 3, 4, 5, 6, 7, 8])

        # Explicitly specifying the ansatz and feature map:
        feature_map = ZZFeatureMap(feature_dimension=num_qubits)
        ansatz = RealAmplitudes(num_qubits=num_qubits)

        qc = QuantumCircuit(num_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        qnn = SamplerQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            interpret=parity,
            output_shape=2
        )

        qnn.forward(input_data=[1, 2], weights=[1, 2, 3, 4, 5, 6, 7, 8])

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
        pass_manager: BasePassManager | None = None,
    ):
        r"""
        Args:
            circuit: The parametrized quantum
                circuit that generates the samples of this network. If a
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` is passed,
                the `input_params` and `weight_params` do not have to be provided, because these two
                properties are taken from the
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit`.
            sampler: The sampler primitive used to compute the neural network's results. If
                ``None`` is given, a default instance of the reference sampler defined by
                :class:`~qiskit.primitives.Sampler` will be used.

                .. warning::

                    The assignment ``sampler=None`` defaults to using
                    :class:`~qiskit.primitives.Sampler`, which points to a deprecated Sampler V1
                    (as of Qiskit 1.2). ``SamplerQNN`` will adopt Sampler V2 as default no later than
                    Qiskit Machine Learning 0.9.

            input_params: The parameters of the circuit corresponding to the input. If a
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` is provided the
                `input_params` value here is ignored. Instead, the value is taken from the
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` input_parameters.
            weight_params: The parameters of the circuit corresponding to the trainable weights. If a
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` is provided the
                `weight_params` value here is ignored. Instead, the value is taken from the
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` ``weight_parameters``.
            sparse: Returns whether the output is sparse or not.
            interpret: A callable that maps the measured integer to another unsigned integer or tuple
                of unsigned integers. These are used as new indices for the (potentially sparse)
                output array. If no interpret function is passed, then an identity function will be
                used by this neural network.
            output_shape: The output shape of the custom interpretation. For SamplerV1, it is ignored
                if no custom interpret method is provided where the shape is taken to be
                ``2^circuit.num_qubits``.
            gradient: An optional sampler gradient to be used for the backward pass. If ``None`` is
                given, a default instance of
                :class:`~qiskit_machine_learning.gradients.ParamShiftSamplerGradient` will be used.
            input_gradients: Determines whether to compute gradients with respect to input data. Note
                that this parameter is ``False`` by default, and must be explicitly set to ``True``
                for a proper gradient computation when using
                :class:`~qiskit_machine_learning.connectors.TorchConnector`.
            pass_manager: The pass manager to transpile the circuits, if necessary.
                Defaults to ``None``, as some primitives do not need transpiled circuits.
        Raises:
            QiskitMachineLearningError: Invalid parameter values.
        """
        # set primitive, provide default
        if sampler is None:
            sampler = Sampler()

        if isinstance(sampler, BaseSamplerV1):
            issue_deprecation_msg(
                msg="V1 Primitives are deprecated",
                version="0.8.0",
                remedy="Use V2 primitives for continued compatibility and support.",
                period="4 months",
            )
        self.sampler = sampler
        if hasattr(circuit.layout, "_input_qubit_count"):
            self.num_virtual_qubits = circuit.layout._input_qubit_count
        else:
            if pass_manager is None:
                self.num_virtual_qubits = circuit.num_qubits
            else:
                circuit = pass_manager.run(circuit)
                self.num_virtual_qubits = circuit.layout._input_qubit_count

        self._org_circuit = circuit

        if isinstance(circuit, QNNCircuit):
            self._input_params = list(circuit.input_parameters)
            self._weight_params = list(circuit.weight_parameters)
        else:
            self._input_params = list(input_params) if input_params is not None else []
            self._weight_params = list(weight_params) if weight_params is not None else []

        if sparse:
            _optionals.HAS_SPARSE.require_now("DOK")

        self.set_interpret(interpret, output_shape)
        # set gradient
        if gradient is None:
            if isinstance(sampler, BaseSamplerV1):
                gradient = ParamShiftSamplerGradient(sampler=self.sampler)
            else:
                if pass_manager is None:
                    logger.warning(
                        "No gradient function provided, creating a gradient function."
                        " If your Sampler requires transpilation, please provide a pass manager."
                    )
                gradient = ParamShiftSamplerGradient(
                    sampler=self.sampler, pass_manager=pass_manager
                )
        self.gradient = gradient

        self._input_gradients = input_gradients

        super().__init__(
            num_inputs=len(self._input_params),
            num_weights=len(self._weight_params),
            sparse=sparse,
            output_shape=self._output_shape,
            input_gradients=self._input_gradients,
        )

        if len(circuit.clbits) == 0:
            circuit = circuit.copy()
            circuit.measure_all()
        self._circuit = self._reparameterize_circuit(circuit, input_params, weight_params)

    @property
    def circuit(self) -> QuantumCircuit:
        """Returns the underlying quantum circuit."""
        return self._org_circuit

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
        """Validate and compute the output shape.
        Raises:
            QiskitMachineLearningError: If no output shape is given.
            QiskitMachineLearningError: If an invalid ``sampler``provided.
        """

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
                    "determined as 2^num_virtual_qubits."
                )
            output_shape_ = (2**self.num_virtual_qubits,)
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
            if isinstance(self.sampler, BaseSamplerV1):
                counts = result.quasi_dists[i]

            elif isinstance(self.sampler, BaseSamplerV2):
                if hasattr(result[i].data, "meas"):
                    bitstring_counts = result[i].data.meas.get_counts()
                else:
                    # Fallback to 'c' if 'meas' is not available.
                    bitstring_counts = result[i].data.c.get_counts()
                # Normalize the counts to probabilities
                total_shots = sum(bitstring_counts.values())
                probabilities = {k: v / total_shots for k, v in bitstring_counts.items()}

                # Convert to quasi-probabilities
                counts = QuasiDistribution(probabilities)
                counts = {k: v for k, v in counts.items() if int(k) < 2**self.num_virtual_qubits}
            else:
                raise QiskitMachineLearningError(
                    "The accepted estimators are BaseSamplerV1 (deprecated) and BaseSamplerV2; "
                    + f"got {type(self.sampler)} instead."
                )
            # evaluate probabilities
            for b, v in counts.items():
                key = self._interpret(b)
                if isinstance(key, Integral):
                    key = (cast(int, key),)
                key = (i, *key)  # type: ignore
                prob[key] += v

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

        if isinstance(self.sampler, BaseSamplerV1):
            job = self.sampler.run([self._circuit] * num_samples, parameter_values)
        elif isinstance(self.sampler, BaseSamplerV2):
            job = self.sampler.run(
                [(self._circuit, parameter_values[i]) for i in range(num_samples)]
            )
        else:
            raise QiskitMachineLearningError(
                "The accepted estimators are BaseSamplerV1 (deprecated) and BaseSamplerV2; "
                + f"got {type(self.sampler)} instead."
            )
        try:
            results = job.result()
        except Exception as exc:
            raise QiskitMachineLearningError(f"Sampler job failed: {exc}") from exc
        result = self._postprocess(num_samples, results)
        return result

    def _backward(
        self,
        input_data: np.ndarray | None,
        weights: np.ndarray | None,
    ) -> tuple[np.ndarray | SparseArray | None, np.ndarray | SparseArray | None]:
        """Backward pass of the network."""
        # prepare parameters in the required format
        parameter_values, num_samples = self._preprocess_forward(input_data, weights)

        input_grad, weights_grad = None, None

        if np.prod(parameter_values.shape) > 0:
            circuits = [self._circuit] * num_samples
            job = None
            if self._input_gradients:
                job = self.gradient.run(circuits, parameter_values)  # type: ignore[arg-type]
            elif len(parameter_values[0]) > self._num_inputs:
                params = [self._circuit.parameters[self._num_inputs :]] * num_samples
                job = self.gradient.run(
                    circuits, parameter_values, parameters=params  # type: ignore[arg-type]
                )

            if job is not None:
                try:
                    results = job.result()
                except Exception as exc:
                    raise QiskitMachineLearningError(f"Sampler job failed: {exc}") from exc

                input_grad, weights_grad = self._postprocess_gradient(num_samples, results)

        return input_grad, weights_grad
