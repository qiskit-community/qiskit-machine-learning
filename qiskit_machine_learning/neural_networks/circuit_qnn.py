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

from typing import Tuple, Union, List, Callable, Any, Optional, Dict

import numpy as np
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
                 interpret: Union[str, Callable[[Tuple[int, ...]], Any]] = 'tuple',
                 dense: bool = False, output_shape: Union[int, Tuple[int, ...]] = None,
                 gradient: Gradient = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None
                 ) -> None:
        """Initializes the Circuit Quantum Neural Network.

        Args:
            circuit: The (parametrized) quantum circuit that generates the samples of this network.
            input_params: The parameters of the circuit corresponding to the input.
            weight_params: The parameters of the circuit corresponding to the trainable weights.
            interpret: Determines the output format, possible choices are:
                * 'tuple' (default): a tuple of binary values, e.g. (0, 1, 0, 1, 0)
                * 'str': a bitstring of type str, e.g. '01010'
                * 'int': an integer corresponding to the bitstring, e.g. 10
                * a custom callable that takes a sample of type 'tuple' and maps it to some other
                output, output should be hashable for sparse representation of probabilities
                and probability gradients.
            dense: Whether to return a dense (array with 'output_shape') or sparse (dict)
                probabilities. Dense probabilities require "interpret == 'int'" where the integer
                will be the index in the array of probabilities.
                TODO: what about "return_samples"??? (cf. base class)
                TODO: update return types to handle dictionaries and arrays
            output_shape: Gives the output_shape in case of a custom interpret callable. If this is
                None, the output_shape is set to 1.
            gradient: The gradient converter to be used for the probability gradients.
            quantum_instance: The quantum instance to evaluate the circuits.

        Raises:
            QiskitMachineLearningError: if an incorrect value for `interpret` or `output_shape`
                is passed.
        """

        # TODO: currently cannot handle statevector simulator, at least throw exception

        # copy circuit and add measurements in case non are given
        self._circuit = circuit.copy()
        if quantum_instance.is_statevector:
            if len(self._circuit.clbits) > 0:
                self._circuit.remove_final_measurements()
        elif len(self._circuit.clbits) == 0:
            self._circuit.measure_all()

        self._input_params = list(input_params or [])
        self._weight_params = list(weight_params or [])
        self._interpret = interpret
        self._dense = dense
        self._gradient = gradient

        if isinstance(quantum_instance, (BaseBackend, Backend)):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance

        # TODO this should not be necessary... but currently prop grads fail otherwise
        from qiskit import Aer
        self._sampler = CircuitSampler(Aer.get_backend('statevector_simulator'), param_qobj=False)

        # construct probability gradient opflow object
        grad_circuit = circuit.copy()
        grad_circuit.remove_final_measurements()  # TODO: ideally this would not be necessary
        params = list(input_params) + list(weight_params)
        self._grad_circuit = Gradient().convert(CircuitStateFn(grad_circuit), params)

        output_shape_: Union[int, Tuple[int, ...]] = -1
        if isinstance(interpret, str):
            if interpret in ('str', 'int'):
                output_shape_ = (quantum_instance.run_config.shots, 1)
            elif interpret == 'tuple':
                output_shape_ = (quantum_instance.run_config.shots, self.circuit.num_qubits)
            else:
                raise QiskitMachineLearningError(f'Unknown interpret string: {interpret}!')
        elif callable(interpret):
            # parameter: output_shape: Union[int, Tuple[int, ...]]
            if output_shape is None:
                output_shape_ = (quantum_instance.run_config.shots, 1)
            else:
                if isinstance(output_shape, int):
                    output_shape_ = (quantum_instance.run_config.shots, output_shape)
                elif isinstance(output_shape, tuple):
                    output_shape_ = (quantum_instance.run_config.shots, *output_shape)
                else:
                    raise QiskitMachineLearningError(
                        f'Unsupported output_shape type: {interpret}!')
        else:
            raise QiskitMachineLearningError(f'Unsupported interpret value: {interpret}!')

        super().__init__(len(self._input_params), len(self._weight_params), output_shape_)

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
        """Returns the quantum instance to evaluate the circuits."""
        return self._quantum_instance

    @property
    def interpret(self) -> Union[str, Callable[[Tuple[int, ...]], Any]]:
        """Returns the interpret option (str) or callable."""
        return self._interpret

    def _interpret_bitstring(self, bitstr: str):
        """Interprets a measured bitstring and returns the required format."""

        def _bit_string_to_tuple(bitstr: str):
            # pylint:disable=consider-using-generator
            return tuple([1 if char == '1' else 0 for char in bitstr])

        if isinstance(self._interpret, str):
            if self._interpret == 'str':
                return bitstr
            elif self._interpret == 'tuple':
                return _bit_string_to_tuple(bitstr)
            elif self.interpret == 'int':
                return int(bitstr, 2)
        elif callable(self._interpret):
            return self._interpret(_bit_string_to_tuple(bitstr))

    def _sample(self, input_data: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if self._quantum_instance.is_statevector:
            raise QiskitMachineLearningError('Sampling does not work with statevector simulator!')

        # combine parameter dictionary
        param_values = {p: input_data[i] for i, p in enumerate(self.input_params)}
        param_values.update({p: weights[i] for i, p in enumerate(self.weight_params)})

        # evaluate operator
        orig_memory = self.quantum_instance.backend_options.get('memory')
        self.quantum_instance.backend_options['memory'] = True
        result = self.quantum_instance.execute(self.circuit.bind_parameters(param_values))
        self.quantum_instance.backend_options['memory'] = orig_memory

        # return samples
        return np.array([self._interpret_bitstring(b) for b in result.get_memory()])

    def _probabilities(self, input_data: np.ndarray, weights: np.ndarray
                       ) -> Union[np.ndarray, Dict[Any, float]]:
        # todo: batches
        # combine parameter dictionary
        param_values = {p: input_data[i] for i, p in enumerate(self.input_params)}
        param_values.update({p: weights[i] for i, p in enumerate(self.weight_params)})

        # evaluate operator
        result = self.quantum_instance.execute(
            self.circuit.bind_parameters(param_values))
        counts = result.get_counts()
        shots = sum(counts.values())
        prob: Dict[Any, float] = {}
        for b, v in counts.items():
            key = self._interpret_bitstring(b)
            prob[key] = prob.get(key, 0.0) + v / shots

        if self._dense:
            prob_array = np.zeros(self._output_shape)
            for k, prob_value in prob.items():
                prob_array[0, k] = prob_value
            return prob_array
        else:
            return prob

    def _probability_gradients(self, input_data: np.ndarray, weights: np.ndarray
                               ) -> Tuple[Union[np.ndarray, List[Dict]],
                                          Union[np.ndarray, List[Dict]]]:
        # todo: batches
        # combine parameter dictionary
        param_values = {p: input_data[i] for i, p in enumerate(self.input_params)}
        param_values.update({p: weights[i] for i, p in enumerate(self.weight_params)})

        # TODO: additional "bind_parameters" should not be necessary, seems like a bug to be fixed
        grad = self._sampler.convert(self._grad_circuit, param_values
                                     ).bind_parameters(param_values).eval()

        # TODO: map to dictionary to pretend sparse logic --> needs to be fixed in opflow!
        input_grad_dicts: List[Dict] = []
        if self.num_inputs > 0:
            input_grad_dicts = [{} for _ in range(self.num_inputs)]
            for i in range(self.num_inputs):
                for k in range(2 ** self.circuit.num_qubits):
                    key = self._interpret_bitstring(("{:0" + str(self.circuit.num_qubits) + "b}"
                                                     ).format(k))
                    input_grad_dicts[i][key] = (input_grad_dicts[i].get(key, 0.0) +
                                                np.real(grad[i][k]))

        weights_grad_dicts: List[Dict] = []
        if self.num_weights > 0:
            weights_grad_dicts = [{} for _ in range(self.num_weights)]
            for i in range(self.num_weights):
                for k in range(2 ** self.circuit.num_qubits):
                    key = self._interpret_bitstring(("{:0" + str(self.circuit.num_qubits) + "b}"
                                                     ).format(k))
                    weights_grad_dicts[i][key] = (weights_grad_dicts[i].get(key, 0.0) +
                                                  np.real(grad[i + self.num_inputs][k]))

        if self._dense:
            input_grad_array = np.zeros((self.num_inputs, *self.output_shape))
            for i in range(self.num_inputs):
                for k, grad in input_grad_dicts[i].items():
                    input_grad_array[i, 0, k] = grad

            weights_grad_array = np.zeros((self.num_weights, *self.output_shape))
            for i in range(self.num_weights):
                for k, grad in weights_grad_dicts[i].items():
                    weights_grad_array[i, 0, k] = grad

            return input_grad_array, weights_grad_array
        else:
            return input_grad_dicts, weights_grad_dicts
