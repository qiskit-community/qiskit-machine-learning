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

"""A Sampling Neural Network based on a given quantum circuit."""
import logging
from numbers import Integral
from typing import Tuple, Union, List, Callable, Optional, cast, Iterable

import numpy as np

from scipy.sparse import coo_matrix

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import Gradient, CircuitSampler, StateFn, OpflowError, OperatorBase
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance, deprecate_function

import qiskit_machine_learning.optionals as _optionals
from .sampling_neural_network import SamplingNeuralNetwork
from ..exceptions import QiskitMachineLearningError, QiskitError

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


class CircuitQNN(SamplingNeuralNetwork):
    """Pending deprecation: A sampling neural network based on a given quantum circuit."""

    @deprecate_function(
        "The CircuitQNN class has been superseded by the "
        "qiskit_machine_learning.neural_networks.SamplerQNN "
        "This class will be deprecated in a future release and subsequently "
        "removed after that.",
        stacklevel=3,
        category=PendingDeprecationWarning,
    )
    def __init__(
        self,
        circuit: QuantumCircuit,
        input_params: Optional[List[Parameter]] = None,
        weight_params: Optional[List[Parameter]] = None,
        sparse: bool = False,
        sampling: bool = False,
        interpret: Optional[Callable[[int], Union[int, Tuple[int, ...]]]] = None,
        output_shape: Union[int, Tuple[int, ...]] = None,
        gradient: Gradient = None,
        quantum_instance: Optional[Union[QuantumInstance, Backend]] = None,
        input_gradients: bool = False,
    ) -> None:
        """
        Args:
            circuit: The parametrized quantum circuit that generates the samples of this network.
                There will be an attempt to transpile this circuit and cache the transpiled circuit
                for subsequent usages by the network. If for some reasons the circuit can't be
                transpiled, e.g. it originates from
                :class:`~qiskit_machine_learning.circuit.library.RawFeatureVector`, the circuit
                will be transpiled every time it is required to be executed and only when all
                parameters are bound. This may impact overall performance on the network.
            input_params: The parameters of the circuit corresponding to the input.
            weight_params: The parameters of the circuit corresponding to the trainable weights.
            sparse: Returns whether the output is sparse or not.
            sampling: Determines whether the network returns a batch of samples or (possibly
                sparse) array of probabilities in its forward pass. In case of probabilities,
                the backward pass returns the probability gradients, while it returns
                ``(None, None)`` in the case of samples. Note that ``sampling==True`` will always
                result in a dense return array independent of the other settings.
            interpret: A callable that maps the measured integer to another unsigned integer or
                tuple of unsigned integers. These are used as new indices for the (potentially
                sparse) output array. If this is used, and ``sampling==False``, the output shape of
                the output needs to be given as a separate argument. If no interpret function is
                passed, then an identity function will be used by this neural network.
            output_shape: The output shape of the custom interpretation, only used in the case
                where an interpret function is provided and ``sampling==False``. Note that in the
                remaining cases, the output shape is automatically inferred by: ``2^num_qubits`` if
                ``sampling==False`` and ``interpret==None``, ``(num_samples,1)``
                if ``sampling==True`` and ``interpret==None``, and
                ``(num_samples, interpret_shape)`` if ``sampling==True`` and an interpret function
                is provided.
            gradient: The gradient converter to be used for the probability gradients.
            quantum_instance: The quantum instance to evaluate the circuits. Note that
                if ``sampling==True``, a statevector simulator is not a valid backend for the
                quantum instance.
            input_gradients: Determines whether to compute gradients with respect to input data.
                Note that this parameter is ``False`` by default, and must be explicitly set to
                ``True`` for a proper gradient computation when using ``TorchConnector``.
        Raises:
            QiskitMachineLearningError: if ``interpret`` is passed without ``output_shape``.

        """
        self._input_params = list(input_params or [])
        self._weight_params = list(weight_params or [])

        # initialize gradient properties
        self.input_gradients = input_gradients

        sparse = False if sampling else sparse
        if sparse:
            _optionals.HAS_SPARSE.require_now("DOK")

        # copy circuit and add measurements in case non are given
        # TODO: need to be able to handle partial measurements! (partial trace...)
        self._circuit = circuit.copy()
        # we have not transpiled the circuit yet
        self._circuit_transpiled = False
        # these original values may be re-used when a quantum instance is set,
        # but initially it was None
        self._original_output_shape = output_shape
        # next line is required by pylint only
        self._interpret = interpret
        self._original_interpret = interpret

        # we need this property in _set_quantum_instance despite it is initialized
        # in the super class later on, review of SamplingNN is required.
        self._sampling = sampling

        # set quantum instance and derive target output_shape and interpret
        self._set_quantum_instance(quantum_instance, output_shape, interpret)

        # init super class
        super().__init__(
            len(self._input_params),
            len(self._weight_params),
            sparse,
            sampling,
            self._output_shape,
            self._input_gradients,
        )

        self._original_circuit = circuit
        # use given gradient or default
        self._gradient = gradient or Gradient()

    def _construct_gradient_circuit(self):
        if self._gradient_circuit_constructed:
            return

        self._gradient_circuit: OperatorBase = None
        try:
            # todo: avoid copying the circuit
            grad_circuit = self._original_circuit.copy()
            grad_circuit.remove_final_measurements()  # ideally this would not be necessary
            if self._input_gradients:
                params = self._input_params + self._weight_params
            else:
                params = self._weight_params
            self._gradient_circuit = self._gradient.convert(StateFn(grad_circuit), params)
        except (ValueError, TypeError, OpflowError, QiskitError):
            logger.warning("Cannot compute gradient operator! Continuing without gradients!")

        self._gradient_circuit_constructed = True

    def _compute_output_shape(self, interpret, output_shape, sampling) -> Tuple[int, ...]:
        """Validate and compute the output shape."""
        # a safety check cause we use quantum instance below
        if self._quantum_instance is None:
            raise QiskitMachineLearningError(
                "A quantum instance is required to compute output shape!"
            )

        # this definition is required by mypy
        output_shape_: Tuple[int, ...] = (-1,)
        # todo: move sampling code to the super class
        if sampling:
            if output_shape is not None:
                # Warn user that output_shape parameter will be ignored
                logger.warning(
                    "In sampling mode, output_shape will be automatically inferred "
                    "from the number of samples and interpret function, if provided."
                )

            num_samples = self._quantum_instance.run_config.shots
            if interpret is not None:
                ret = interpret(0)  # infer shape from function
                result = np.array(ret)
                if len(result.shape) == 0:
                    output_shape_ = (num_samples, 1)
                else:
                    output_shape_ = (num_samples, *result.shape)
            else:
                output_shape_ = (num_samples, 1)
        else:
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

        # final validation
        output_shape_ = self._validate_output_shape(output_shape_)

        return output_shape_

    @property
    def circuit(self) -> QuantumCircuit:
        """Returns the underlying quantum circuit."""
        return self._circuit

    @property
    def input_params(self) -> List:
        """Returns the list of input parameters."""
        return self._input_params

    @property
    def weight_params(self) -> List:
        """Returns the list of trainable weights parameters."""
        return self._weight_params

    @property
    def interpret(self) -> Optional[Callable[[int], Union[int, Tuple[int, ...]]]]:
        """Returns interpret function to be used by the neural network. If it is not set in
        the constructor or can not be implicitly derived (e.g. a quantum instance is not provided),
        then ``None`` is returned."""
        return self._interpret

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Returns the quantum instance to evaluate the circuit."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Optional[Union[QuantumInstance, Backend]]) -> None:
        """Sets the quantum instance to evaluate the circuit and make sure circuit has
        measurements or not depending on the type of backend used.
        """
        self._set_quantum_instance(
            quantum_instance, self._original_output_shape, self._original_interpret
        )

    def _set_quantum_instance(
        self,
        quantum_instance: Optional[Union[QuantumInstance, Backend]],
        output_shape: Union[int, Tuple[int, ...]],
        interpret: Optional[Callable[[int], Union[int, Tuple[int, ...]]]],
    ) -> None:
        """
        Internal method to set a quantum instance and compute/initialize internal properties such
        as an interpret function, output shape and a sampler.

        Args:
            quantum_instance: A quantum instance to set.
            output_shape: An output shape of the custom interpretation.
            interpret: A callable that maps the measured integer to another unsigned integer or
                tuple of unsigned integers.
        """
        if isinstance(quantum_instance, Backend):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance

        if self._quantum_instance is not None:
            # add measurements in case none are given
            if self._quantum_instance.is_statevector:
                if len(self._circuit.clbits) > 0:
                    self._circuit.remove_final_measurements()
            elif len(self._circuit.clbits) == 0:
                self._circuit.measure_all()

            # set interpret and compute output shape
            self.set_interpret(interpret, output_shape)

            # prepare sampler
            self._sampler = CircuitSampler(self._quantum_instance, param_qobj=False, caching="all")

            # transpile the QNN circuit
            try:
                self._circuit = self._quantum_instance.transpile(
                    self._circuit, pass_manager=self._quantum_instance.unbound_pass_manager
                )[0]
                self._circuit_transpiled = True
            except QiskitError:
                # likely it is caused by RawFeatureVector, we just ignore this error and
                # transpile circuits when it is required.
                self._circuit_transpiled = False
        else:
            self._output_shape = output_shape

    @property
    def input_gradients(self) -> bool:
        """Returns whether gradients with respect to input data are computed by this neural network
        in the ``backward`` method or not. By default such gradients are not computed."""
        return self._input_gradients

    @input_gradients.setter
    def input_gradients(self, input_gradients: bool) -> None:
        """Turn on/off gradient with respect to input data."""
        self._input_gradients = input_gradients

        # reset gradient circuit
        self._gradient_circuit = None
        self._gradient_circuit_constructed = False

    def set_interpret(
        self,
        interpret: Optional[Callable[[int], Union[int, Tuple[int, ...]]]],
        output_shape: Union[int, Tuple[int, ...]] = None,
    ) -> None:
        """Change 'interpret' and corresponding 'output_shape'. If self.sampling==True, the
        output _shape does not have to be set and is inferred from the interpret function.
        Otherwise, the output_shape needs to be given.

        Args:
            interpret: A callable that maps the measured integer to another unsigned integer or
                tuple of unsigned integers. See constructor for more details.
            output_shape: The output shape of the custom interpretation, only used in the case
                where an interpret function is provided and ``sampling==False``. See constructor
                for more details.
        """

        # save original values
        self._original_output_shape = output_shape
        self._original_interpret = interpret

        # derive target values to be used in computations
        self._output_shape = self._compute_output_shape(interpret, output_shape, self._sampling)
        self._interpret = interpret if interpret is not None else lambda x: x

    def _sample(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> np.ndarray:
        self._check_quantum_instance("samples")

        if self._quantum_instance.is_statevector:
            raise QiskitMachineLearningError("Sampling does not work with statevector simulator!")

        # evaluate operator
        orig_memory = self._quantum_instance.backend_options.get("memory")
        self._quantum_instance.backend_options["memory"] = True

        circuits = []
        # iterate over samples, each sample is an element of a batch
        num_samples = input_data.shape[0]
        for i in range(num_samples):
            param_values = {
                input_param: input_data[i, j] for j, input_param in enumerate(self._input_params)
            }
            param_values.update(
                {weight_param: weights[j] for j, weight_param in enumerate(self._weight_params)}
            )
            circuits.append(self._circuit.bind_parameters(param_values))

        if self._quantum_instance.bound_pass_manager is not None:
            circuits = self._quantum_instance.transpile(
                circuits, pass_manager=self._quantum_instance.bound_pass_manager
            )

        result = self._quantum_instance.execute(circuits, had_transpiled=self._circuit_transpiled)
        # set the memory setting back
        self._quantum_instance.backend_options["memory"] = orig_memory

        # return samples
        samples = np.zeros((num_samples, *self._output_shape))
        # collect them from all executed circuits
        for i, circuit in enumerate(circuits):
            memory = result.get_memory(circuit)
            for j, b in enumerate(memory):
                samples[i, j, :] = self._interpret(int(b, 2))
        return samples

    def _probabilities(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Union[np.ndarray, SparseArray]:
        self._check_quantum_instance("probabilities")

        # evaluate operator
        circuits = []
        num_samples = input_data.shape[0]
        for i in range(num_samples):
            param_values = {
                input_param: input_data[i, j] for j, input_param in enumerate(self._input_params)
            }
            param_values.update(
                {weight_param: weights[j] for j, weight_param in enumerate(self._weight_params)}
            )
            circuits.append(self._circuit.bind_parameters(param_values))

        if self._quantum_instance.bound_pass_manager is not None:
            circuits = self._quantum_instance.transpile(
                circuits, pass_manager=self._quantum_instance.bound_pass_manager
            )

        result = self._quantum_instance.execute(circuits, had_transpiled=self._circuit_transpiled)
        # initialize probabilities
        if self._sparse:
            # pylint: disable=import-error
            from sparse import DOK

            prob = DOK((num_samples, *self._output_shape))
        else:
            prob = np.zeros((num_samples, *self._output_shape))

        for i, circuit in enumerate(circuits):
            counts = result.get_counts(circuit)
            shots = sum(counts.values())

            # evaluate probabilities
            for b, v in counts.items():
                key = self._interpret(int(b, 2))
                if isinstance(key, Integral):
                    key = (cast(int, key),)
                key = (i, *key)  # type: ignore
                prob[key] += v / shots

        if self._sparse:
            return prob.to_coo()
        else:
            return prob

    def _probability_gradients(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Tuple[Union[np.ndarray, SparseArray], Union[np.ndarray, SparseArray]]:
        self._check_quantum_instance("probability gradients")

        self._construct_gradient_circuit()

        # check whether gradient circuit could be constructed
        if self._gradient_circuit is None:
            return None, None

        num_samples = input_data.shape[0]

        # initialize empty gradients
        input_grad = None  # by default we don't have data gradients
        if self._sparse:
            # pylint: disable=import-error
            from sparse import DOK

            if self._input_gradients:
                input_grad = DOK((num_samples, *self._output_shape, self._num_inputs))
            weights_grad = DOK((num_samples, *self._output_shape, self._num_weights))
        else:
            if self._input_gradients:
                input_grad = np.zeros((num_samples, *self._output_shape, self._num_inputs))
            weights_grad = np.zeros((num_samples, *self._output_shape, self._num_weights))

        param_values = {
            input_param: input_data[:, j] for j, input_param in enumerate(self._input_params)
        }
        param_values.update(
            {
                weight_param: np.full(num_samples, weights[j])
                for j, weight_param in enumerate(self._weight_params)
            }
        )

        converted_op = self._sampler.convert(self._gradient_circuit, param_values)
        # if statement is a workaround for https://github.com/Qiskit/qiskit-terra/issues/7608
        if len(converted_op.parameters) > 0:
            # create an list of parameter bindings, each element corresponds to a sample in the dataset
            param_bindings = [
                {param: param_values[i] for param, param_values in param_values.items()}
                for i in range(num_samples)
            ]

            grad = []
            # iterate over gradient vectors and bind the correct leftover parameters
            for g_i, param_i in zip(converted_op, param_bindings):
                # bind or re-bind remaining values and evaluate the gradient
                grad.append(g_i.bind_parameters(param_i).eval())
        else:
            grad = converted_op.eval()

        if self._input_gradients:
            num_grad_vars = self._num_inputs + self._num_weights
        else:
            num_grad_vars = self._num_weights

        # construct gradients
        for sample in range(num_samples):
            for i in range(num_grad_vars):
                coo_grad = coo_matrix(grad[sample][i])  # this works for sparse and dense case

                # get index for input or weights gradients
                if self._input_gradients:
                    grad_index = i if i < self._num_inputs else i - self._num_inputs
                else:
                    grad_index = i

                for _, k, val in zip(coo_grad.row, coo_grad.col, coo_grad.data):
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
        # end of for each sample

        if self._sparse:
            if self._input_gradients:
                input_grad = input_grad.to_coo()
            weights_grad = weights_grad.to_coo()

        return input_grad, weights_grad

    def _check_quantum_instance(self, feature: str):
        if self._quantum_instance is None:
            raise QiskitMachineLearningError(
                f"Evaluation of {feature} requires a quantum instance!"
            )
