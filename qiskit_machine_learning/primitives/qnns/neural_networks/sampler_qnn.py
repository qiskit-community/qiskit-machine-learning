import logging
from numbers import Integral
import numpy as np
from typing import Optional, Union, List, Tuple, Callable, cast, Iterable
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.exceptions import QiskitMachineLearningError, QiskitError

logger = logging.getLogger(__name__)

# from primitives.gradient import FiniteDiffEstimatorGradient, FiniteDiffSamplerGradient

from scipy.sparse import coo_matrix

from primitives.gradient.param_shift_sampler_gradient import ParamShiftSamplerGradient
from qiskit.primitives import Sampler

class SamplerQNN():

    def __init__(
            self,
            circuit: QuantumCircuit,
            input_params: Optional[List[Parameter]] = None,
            weight_params: Optional[List[Parameter]] = None,
            interpret: Optional[Callable[[int], Union[int, Tuple[int, ...]]]] = None,
            output_shape: Union[int, Tuple[int, ...]] = None,
            sampler_factory: Callable = None,
            gradient_method: str = "param_shift",

    ):
        # IGNORING SPARSE
        # SKIPPING CUSTOM GRADIENT
        # SKIPPING "INPUT GRADIENTS" -> by default with primitives?

        # we allow only one circuit at this moment
        self._circuit = circuit
        # self._gradient = ParamShiftSamplerGradient(sampler, self._circuit)

        self._gradient_method = gradient_method
        self._sampler_factory = sampler_factory

        self._input_params = list(input_params or [])
        self._weight_params = list(weight_params or [])

        self.output_shape = None
        self._num_inputs = len(self._input_params)
        self._num_weights = len(self._weight_params)
        self.num_weights = self._num_weights
        # the circuit must always have measurements.... (?)
        # add measurements in case none are given
        if len(self._circuit.clbits) == 0:
            self._circuit.measure_all()

        self._interpret = interpret
        self._original_interpret = interpret

        # set interpret and compute output shape
        self.set_interpret(interpret, output_shape)

        self._input_gradients = None

    # def output_shape(self):
    #     return self._output_shape
    #     return self._output_shape

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc_info):
        self.close()

    def open(self):
        # we should delay instantiation of the primitives till they are really required
        if self._gradient_method == "param_shift":
            # if gradient method -> sampler with gradient functionality
            self._sampler = ParamShiftSamplerGradient(
                circuit = self._circuit,
                sampler = self._sampler_factory
            )
        else:
            # if no gradient method -> sampler without gradient functionality
            self._sampler = self._sampler_factory(
                circuits = [self._circuit],
                parameters = [self._input_params + self._weight_params]
            )
        pass

    def close(self):
        self._sampler.__exit__()

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
        self._output_shape = self._compute_output_shape(interpret, output_shape)
        self._interpret = interpret if interpret is not None else lambda x: x
        self.output_shape = self._output_shape

    def _compute_output_shape(self, interpret, output_shape) -> Tuple[int, ...]:
        """Validate and compute the output shape."""

        # this definition is required by mypy
        output_shape_: Tuple[int, ...] = (-1,)
        # todo: move sampling code to the super class

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

            output_shape_ = (2 ** self._circuit.num_qubits,)

        # # final validation
        # output_shape_ = self._validate_output_shape(output_shape_)

        return output_shape_

    def forward(
            self,
            input_data: Optional[Union[List[float], np.ndarray, float]],
            weights: Optional[Union[List[float], np.ndarray, float]],
    ) -> np.ndarray:

        result = self._forward(input_data, weights)
        return result

    def _preprocess(self, input_data, weights):
        if len(input_data.shape) == 1:
            input_data = np.expand_dims(input_data, 0)
        num_samples = input_data.shape[0]
        # quick fix for 0 inputs
        if num_samples == 0:
            num_samples = 1

        parameters = []
        for i in range(num_samples):
            param_values = [input_data[i,j] for j, input_param in enumerate(self._input_params)]
            param_values += [weights[j] for j, weight_param in enumerate(self._weight_params)]
            parameters.append(param_values)

        return parameters, num_samples

    def _postprocess(self, num_samples, result):

        prob = np.zeros((num_samples, *self._output_shape))

        for i in range(num_samples):
            counts = result[i].quasi_dists[0]
            print(counts)
            shots = sum(counts.values())

            # evaluate probabilities
            for b, v in counts.items():
                key = (i, int(self._interpret(b))) # type: ignore
                prob[key] += v / shots

        return prob

    def _forward(
            self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> np.ndarray:

        parameter_values, num_samples = self._preprocess(input_data, weights)

        # result = self._sampler([0] * num_samples, parameter_values)

        results = []
        for sample in range(num_samples):
            result = self._sampler(parameter_values)
            results.append(result)

        result = self._postprocess(num_samples, results)

        return result

    def backward(
            self,
            input_data: Optional[Union[List[float], np.ndarray, float]],
            weights: Optional[Union[List[float], np.ndarray, float]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],]:

        result = self._backward(input_data, weights)
        return result

    def _preprocess_gradient(self, input_data, weights):

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

        # prepare parameters in the required format
        parameter_values, num_samples = self._preprocess_gradient(input_data, weights)

        results = []
        for sample in range(num_samples):
            if self._input_gradients:
                result = self._sampler.gradient(parameter_values[sample])
            else:
                result = self._sampler.gradient(parameter_values[sample],
                                                 partial=self._sampler._circuit.parameters[self._num_inputs:])

            results.append(result)
        input_grad, weights_grad = self._postprocess_gradient(num_samples, results)

        return None , weights_grad  # `None` for gradients wrt input data, see TorchConnector










