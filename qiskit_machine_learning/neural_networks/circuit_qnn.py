# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A Sampling Neural Network based on a given quantum circuit."""

from numbers import Integral
from typing import (Tuple, Union, List,
                    Callable, Optional, Dict, cast, Iterable)

import numpy as np
from sparse import SparseArray, DOK
from scipy.sparse import coo_matrix

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import Gradient, CircuitSampler, CircuitStateFn
from qiskit.providers import BaseBackend, Backend
from qiskit.utils import QuantumInstance

from .sampling_neural_network import SamplingNeuralNetwork
from ..exceptions import QiskitMachineLearningError


class CircuitQNN(SamplingNeuralNetwork):
    """A Sampling Neural Network based on a given quantum circuit."""

    def __init__(self, circuit: QuantumCircuit,
                 input_params: Optional[List[Parameter]] = None,
                 weight_params: Optional[List[Parameter]] = None,
                 sparse: bool = False,
                 sampling: bool = False,
                 interpret: Optional[Callable[[int], Union[int, Tuple[int, ...]]]] = None,
                 output_shape: Union[int, Tuple[int, ...]] = None,
                 gradient: Gradient = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None
                 ) -> None:
        """Initializes the Circuit Quantum Neural Network.

        Args:
            circuit: The parametrized quantum circuit that generates the samples of this network.
            input_params: The parameters of the circuit corresponding to the input.
            weight_params: The parameters of the circuit corresponding to the trainable weights.
            sparse: Returns whether the output is sparse or not.
            sampling: Determines whether the network returns a batch of samples or (possibly
                sparse) array of probabilities in its forward pass. In case of probabilities,
                the backward pass returns the probability gradients, while it returns (None, None)
                in the case of samples. Note that sampling==True will always result in a
                dense return array independent of the other settings.
            interpret: A callable that maps the measured integer to another unsigned integer or
                tuple of unsigned integers. These are used as new indices for the (potentially
                sparse) output array. If this is used, the output shape of the output needs to be
                given as a separate argument.
            output_shape: The output shape of the custom interpretation. The output shape is
                automatically determined in case of sampling==True.
            gradient: The gradient converter to be used for the probability gradients.
            quantum_instance: The quantum instance to evaluate the circuits.

        Raises:
            QiskitMachineLearningError: if `interpret` is passed without `output_shape`.
        """

        # TODO: need to handle case without a quantum instance
        # TODO: need to be able to handle partial measurements! (partial trace...)
        # copy circuit and add measurements in case non are given
        self._circuit = circuit.copy()
        if quantum_instance.is_statevector:
            if len(self._circuit.clbits) > 0:
                self._circuit.remove_final_measurements()
        elif len(self._circuit.clbits) == 0:
            self._circuit.measure_all()

        self._input_params = list(input_params or [])
        self._weight_params = list(weight_params or [])
        self._interpret = interpret if interpret else lambda x: x
        sparse_ = sparse
        # this definition is required by mypy
        output_shape_: Union[int, Tuple[int, ...]] = -1
        if sampling:
            num_samples = quantum_instance.run_config.shots
            sparse_ = False
            # infer shape from function
            ret = self._interpret(0)
            result = np.array(ret)
            output_shape_ = (num_samples, *result.shape)
            if len(result.shape) == 0:
                output_shape_ = (num_samples, 1)
        else:
            if interpret:
                if output_shape is None:
                    raise QiskitMachineLearningError(
                        'No output shape given, but required in case of custom interpret!')
                output_shape_ = output_shape
            else:
                output_shape_ = (2 ** circuit.num_qubits,)

        self._gradient = gradient

        if isinstance(quantum_instance, (BaseBackend, Backend)):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance
        self._sampler = CircuitSampler(quantum_instance, param_qobj=False, caching='all')

        # construct probability gradient opflow object
        grad_circuit = circuit.copy()
        grad_circuit.remove_final_measurements()  # ideally this would not be necessary
        params = list(input_params) + list(weight_params)
        self._grad_circuit = Gradient().convert(CircuitStateFn(grad_circuit), params)

        super().__init__(len(self._input_params), len(self._weight_params), sparse_, sampling,
                         output_shape_)

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
    def quantum_instance(self) -> QuantumInstance:
        """Returns the quantum instance to evaluate the circuit."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance) -> None:
        """Sets the quantum instance to evaluate the circuit and make sure circuit has
        measurements or not depending on the type of backend used.
        """
        self._quantum_instance = quantum_instance

        # add measurements in case non are given
        if quantum_instance.is_statevector:
            if len(self._circuit.clbits) > 0:
                self._circuit.remove_final_measurements()
        elif len(self._circuit.clbits) == 0:
            self._circuit.measure_all()

    def set_interpret(self, interpret, output_shape=None):
        """ Change 'interpret' and corresponding 'output_shape'. If self.sampling==True, the
        output _shape does not have to be set and is inferred from the interpret function. Otherwise,
        the output_shape needs to be given."""

        if self.sampling:
            num_samples = self.quantum_instance.run_config.shots

            # infer shape from function
            ret = interpret(0)
            result = np.array(ret)
            output_shape = (num_samples, *result.shape)
            if len(result.shape) == 0:
                output_shape = (num_samples, 1)
            self._output_shape = output_shape
        else:
            if output_shape is None:
                raise QiskitMachineLearningError(
                    'No output shape given, but required in case of custom interpret!')
            elif isinstance(output_shape, Integral):
                output_shape = (output_shape,)
            self._output_shape = output_shape

    def _sample(self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
                ) -> np.ndarray:
    def _sample(self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
                ) -> np.ndarray:
        if self._quantum_instance.is_statevector:
            raise QiskitMachineLearningError('Sampling does not work with statevector simulator!')

        # evaluate operator
        orig_memory = self.quantum_instance.backend_options.get('memory')
        self.quantum_instance.backend_options['memory'] = True

        circuits = []
        # iterate over rows, each row is an element of a batch
        rows = input_data.shape[0]
        for i in range(rows):
            param_values = {input_param: input_data[i, j]
                            for j, input_param in enumerate(self.input_params)}
            param_values.update({weight_param: weights[j]
                                 for j, weight_param in enumerate(self.weight_params)})
            circuits.append(self._circuit.bind_parameters(param_values))

        result = self._quantum_instance.execute(circuits)
        # set the memory setting back
        self.quantum_instance.backend_options['memory'] = orig_memory

        # return samples
        samples = np.zeros((rows, *self.output_shape))
        # collect them from all executed circuits
        for i, circuit in enumerate(circuits):
            memory = result.get_memory(circuit)
            for j, b in enumerate(memory):
                samples[i, j, :] = self._interpret(int(b, 2))
        return samples

    def _probabilities(self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
                       ) -> Union[np.ndarray, SparseArray]:
        # evaluate operator
        circuits = []
        rows = input_data.shape[0]
        for i in range(rows):
            param_values = {input_param: input_data[i, j]
                            for j, input_param in enumerate(self.input_params)}
            param_values.update({weight_param: weights[j]
                                 for j, weight_param in enumerate(self.weight_params)})
            circuits.append(self._circuit.bind_parameters(param_values))

        result = self.quantum_instance.execute(circuits)
        # initialize probabilities
        if self.sparse:
            prob = DOK((rows, *self.output_shape))
        else:
            prob = np.zeros((rows, *self.output_shape))

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

        if self.sparse:
            return prob.to_coo()
        else:
            return prob

    def _probability_gradients(self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
                               ) -> Tuple[Union[np.ndarray, SparseArray],
                                          Union[np.ndarray, SparseArray]]:
        rows = input_data.shape[0]

        # initialize empty gradients
        if self._sparse:
            if self.num_inputs > 0:
                input_grad = DOK((rows, *self.output_shape, self.num_inputs))
            else:
                input_grad = np.zeros((rows, *self.output_shape, self.num_inputs))
            if self.num_weights > 0:
                weights_grad = DOK((rows, *self.output_shape, self.num_weights))
            else:
                weights_grad = np.zeros((rows, *self.output_shape, self.num_weights))
        else:
            input_grad = np.zeros((rows, *self.output_shape, self.num_inputs))
            weights_grad = np.zeros((rows, *self.output_shape, self.num_weights))

        for row in range(rows):
            param_values = {input_param: input_data[row, j]
                            for j, input_param in enumerate(self.input_params)}
            param_values.update({weight_param: weights[j]
                                 for j, weight_param in enumerate(self.weight_params)})

            # TODO: additional "bind_parameters" should not be necessary,
            #  seems like a bug to be fixed
            grad = self._sampler.convert(self._grad_circuit, param_values
                                         ).bind_parameters(param_values).eval()

            # construct gradients
            for i in range(self.num_inputs + self.num_weights):
                coo_grad = coo_matrix(grad[i])  # this works for sparse and dense case

                # get index for input or weights gradients
                j = i if i < self.num_inputs else i - self.num_inputs

                for _, k, val in zip(coo_grad.row, coo_grad.col, coo_grad.data):

                    # interpret integer and construct key
                    key = self._interpret(k)
                    if isinstance(key, Integral):
                        key = (row, int(key), j)
                    else:
                        # if key is an array-type, cast to hashable tuple
                        key = tuple(cast(Iterable[int], key))
                        key = (row, *key, j)  # type: ignore

                    # store value for inputs or weights gradients
                    if i < self.num_inputs:
                        input_grad[key] += np.real(val)
                    else:
                        weights_grad[key] += np.real(val)

        if self.sparse:
            return input_grad.to_coo(), weights_grad.to_coo()
        else:
            return input_grad, weights_grad
