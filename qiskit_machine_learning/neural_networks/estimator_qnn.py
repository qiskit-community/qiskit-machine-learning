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

"""An Opflow Quantum Neural Network that allows to use a parametrized opflow object as a
neural network."""
import logging
from typing import List, Optional, Union, Tuple, Dict

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit, ParameterExpression
from qiskit.algorithms.gradients import BaseEstimatorGradient
from qiskit.opflow import (
    Gradient,
    CircuitSampler,
    ListOp,
    OperatorBase,
    ExpectationBase,
    OpflowError,
    ComposedOp,
)
from qiskit.primitives import BaseEstimator
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance
from qiskit.utils.backend_utils import is_aer_provider
import qiskit_machine_learning.optionals as _optionals
from .neural_network import NeuralNetwork
from ..exceptions import QiskitMachineLearningError, QiskitError

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli

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
    """A Neural Network implementation based on the Sampler primitive."""

    def __init__(
        self,
        estimator: BaseEstimator,
        circuit: QuantumCircuit,
        operator: OperatorBase,
        input_params: Optional[List[Parameter]] = None,
        weight_params: Optional[List[Parameter]] = None,
        gradient: Optional[BaseEstimatorGradient] = None,
        input_gradients: bool = False,
    ):
        """
        Args:
            operator: The parametrized operator that represents the neural network.
            input_params: The operator parameters that correspond to the input of the network.
            weight_params: The operator parameters that correspond to the trainable weights.
            exp_val: The Expected Value converter to be used for the operator.
            gradient: The Gradient converter to be used for the operator's backward pass.
            quantum_instance: The quantum instance to evaluate the network.
            input_gradients: Determines whether to compute gradients with respect to input data.
                Note that this parameter is ``False`` by default, and must be explicitly set to
                ``True`` for a proper gradient computation when using ``TorchConnector``.
        """
        self._estimator = estimator
        self._circuit = circuit
        self._operator = operator
        self._input_params = list(input_params) or []
        self._weight_params = list(weight_params) or []
        self._gradient = gradient

        # initialize gradient properties
        self.input_gradients = input_gradients

        output_shape = self._compute_output_shape(operator)
        super().__init__(
            len(self._input_params),
            len(self._weight_params),
            sparse=False,
            output_shape=output_shape,
            input_gradients=input_gradients,
        )

    @property
    def operator(self):
        """Returns the underlying operator of this QNN."""
        return self._operator

    @property
    def input_gradients(self) -> bool:
        """Returns whether gradients with respect to input data are computed by this neural network
        in the ``backward`` method or not. By default such gradients are not computed."""
        return self._input_gradients

    @input_gradients.setter
    def input_gradients(self, input_gradients: bool) -> None:
        """Turn on/off computation of gradients with respect to input data."""
        self._input_gradients = input_gradients

        # reset gradient operator
        self._gradient_operator = None
        self._gradient_operator_constructed = False

    def _compute_output_shape(self, op: OperatorBase) -> Tuple[int, ...]:
        """Determines the output shape of a given operator."""



        # TODO: the whole method should eventually be moved to opflow and rewritten in a better way.
        # if the operator is a composed one, then we only need to look at the first element of it.
        if isinstance(op, ComposedOp):
            return self._compute_output_shape(op.oplist[0].primitive)
        # this "if" statement is on purpose, to prevent sub-classes.
        # pylint:disable=unidiomatic-typecheck
        if type(op) == ListOp:
            shapes = [self._compute_output_shape(op_) for op_ in op.oplist]
            if not np.all([shape == shapes[0] for shape in shapes]):
                raise QiskitMachineLearningError(
                    "Only supports ListOps with children that return the same shape."
                )
            if shapes[0] == (1,):
                out = op.combo_fn(np.zeros((len(op.oplist))))
            else:
                out = op.combo_fn(np.zeros((len(op.oplist), *shapes[0])))
            return out.shape
        else:
            return (1,)

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

    def _postprocess(self, num_samples, results):
        """Post-processing during forward pass of the network."""
        res = np.zeros((num_samples, 1))
        for i in range(num_samples):
            res[i, 0] = results.values[i]
        return res


    def _forward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Union[np.ndarray, SparseArray]:
        # combine parameter dictionary
        # take i-th column as values for the i-th param in a batch
        parameter_values, num_samples = self._preprocess(input_data, weights)

        print(f'parameter_values: {parameter_values}')

        job = self._estimator.run([self._circuit]*num_samples, [self._operator]*num_samples, parameter_values)
        results = job.result()
        return self._postprocess(num_samples, results)


    # def _preprocess_gradient(self, input_data, weights):
    #     """
    #     Pre-processing during backward pass of the network.
    #     """
    #     if len(input_data.shape) == 1:
    #         input_data = np.expand_dims(input_data, 0)

    #     num_samples = input_data.shape[0]
    #     # quick fix for 0 inputs
    #     if num_samples == 0:
    #         num_samples = 1

    #     parameters = []
    #     for i in range(num_samples):
    #         param_values = [input_data[i, j] for j, input_param in enumerate(self._input_params)]
    #         param_values += [weights[j] for j, weight_param in enumerate(self._weight_params)]
    #         parameters.append(param_values)

    #     return parameters, num_samples

    def _postprocess_gradient(self, num_samples, results):
        """
        Post-processing during backward pass of the network.
        """
        input_grad = np.zeros((num_samples, *self._output_shape, self._num_inputs)) if self._input_gradients else None
        weights_grad = np.zeros((num_samples, *self._output_shape, self._num_weights))

        if self._input_gradients:
            num_grad_vars = self._num_inputs + self._num_weights
        else:
            num_grad_vars = self._num_weights

        for sample in range(num_samples):
            for i in range(num_grad_vars):
                if self._input_gradients:
                    if i < self._num_inputs:
                        input_grad[sample, 0, i] = results.values[sample][i]
                    else:
                        weights_grad[sample, 0, i - self._num_inputs] = results.values[sample][i]
                else:
                    weights_grad[sample, 0, i] = results.values[sample][i]

        return input_grad, weights_grad

    def _backward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],]:

        """Backward pass of the network.
        """
        # prepare parameters in the required format
        parameter_values, num_samples = self._preprocess(input_data, weights)
        print(parameter_values)
        if self._input_gradients:
            job = self._gradient.run([self._circuit]*num_samples, [self._operator]*num_samples, parameter_values)
        else:
            job = self._gradient.run([self._circuit]*num_samples, [self._operator]*num_samples, parameter_values,
                                    parameters=[self._circuit.parameters[self._num_inputs:]] *num_samples)

        results = job.result()
        print(results)

        # input_grad, weights_grad = self._postprocess_gradient(num_samples, results)

        # return input_grad, weights_grad  # `None` for gradients wrt input data, see TorchConnector

def _init_observable(observable: BaseOperator | PauliSumOp) -> SparsePauliOp:
    """Initialize observable by converting the input to a :class:`~qiskit.quantum_info.SparsePauliOp`.

    Args:
        observable: The observable.

    Returns:
        The observable as :class:`~qiskit.quantum_info.SparsePauliOp`.

    Raises:
        TypeError: If the observable is a :class:`~qiskit.opflow.PauliSumOp` and has a parameterized
            coefficient.
    """
    if isinstance(observable, SparsePauliOp):
        return observable
    elif isinstance(observable, PauliSumOp):
        if isinstance(observable.coeff, ParameterExpression):
            raise TypeError(
                f"Observable must have numerical coefficient, not {type(observable.coeff)}."
            )
        return observable.coeff * observable.primitive
    elif isinstance(observable, BasePauli):
        return SparsePauliOp(observable)
    elif isinstance(observable, BaseOperator):
        return SparsePauliOp.from_operator(observable)
    elif isinstance(observable, ListOp):
        #TODO: ListOP? or simply use list of operators
        pass
    else:
        return SparsePauliOp(observable)




    # def _backward(
    #     self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    # ) -> Tuple[Optional[Union[np.ndarray, SparseArray]], Optional[Union[np.ndarray, SparseArray]],]:

    #     self._construct_gradient_operator()

    #     # check whether gradient circuit could be constructed
    #     if self._gradient_operator is None:
    #         return None, None

    #     num_samples = input_data.shape[0]
    #     if self._input_gradients:
    #         num_params = self._num_inputs + self._num_weights
    #     else:
    #         num_params = self._num_weights

    #     param_values = {
    #         input_param: input_data[:, j] for j, input_param in enumerate(self._input_params)
    #     }
    #     param_values.update(
    #         {
    #             weight_param: np.full(num_samples, weights[j])
    #             for j, weight_param in enumerate(self._weight_params)
    #         }
    #     )

    #     if self._circuit_sampler:
    #         converted_op = self._circuit_sampler.convert(self._gradient_operator, param_values)
    #         # if statement is a workaround for https://github.com/Qiskit/qiskit-terra/issues/7608
    #         if len(converted_op.parameters) > 0:
    #             # rebind the leftover parameters and evaluate the gradient
    #             grad = self._evaluate_operator(converted_op, num_samples, param_values)
    #         else:
    #             # all parameters are bound by CircuitSampler, so we evaluate the operator directly
    #             grad = np.asarray(converted_op.eval())
    #     else:
    #         # we evaluate gradient operator for each sample separately, so we create a list of operators.
    #         grad = self._evaluate_operator(
    #             [self._gradient_operator] * num_samples, num_samples, param_values
    #         )

    #     grad = np.real(grad)

    #     # this is another workaround to fix output shape of the invocation result of CircuitSampler
    #     if self._output_shape == (1,):
    #         # at least 3 dimensions: batch, output, num_parameters, but in this case we don't have
    #         # output dimension, so we add a dimension that corresponds to the output
    #         grad = grad.reshape((num_samples, 1, num_params))
    #     else:
    #         # swap last axis that corresponds to parameters and axes correspond to the output shape
    #         last_axis = len(grad.shape) - 1
    #         grad = grad.transpose([0, last_axis, *(range(1, last_axis))])

    #     # split into and return input and weights gradients
    #     if self._input_gradients:
    #         input_grad = grad[:, :, : self._num_inputs].reshape(
    #             -1, *self._output_shape, self._num_inputs
    #         )

    #         weights_grad = grad[:, :, self._num_inputs :].reshape(
    #             -1, *self._output_shape, self._num_weights
    #         )
    #     else:
    #         input_grad = None
    #         weights_grad = grad.reshape(-1, *self._output_shape, self._num_weights)

    #     return input_grad, weights_grad

    # def _evaluate_operator(
    #     self,
    #     operator: Union[OperatorBase, List[OperatorBase]],
    #     num_samples: int,
    #     param_values: Dict[Parameter, np.ndarray],
    # ) -> np.ndarray:
    #     """
    #     Evaluates an operator or a list of operators for the samples in the dataset. If an operator
    #     is passed then it is considered as an iterable that has `num_samples` elements. Usually such
    #     operators are obtained as an output from `CircuitSampler`. If a list of operators is passed
    #     then each operator in this list is evaluated with a set of values/parameters corresponding
    #     to the sample index in the `param_values` as the operator in the list.

    #     Args:
    #         operator: operator or list of operators to evaluate.
    #         num_samples: a total number of samples
    #         param_values: parameter values to use for operator evaluation.

    #     Returns:
    #         the result of operator evaluation as an array.
    #     """
    #     # create an list of parameter bindings, each element corresponds to a sample in the dataset
    #     param_bindings = [
    #         {param: param_values[i] for param, param_values in param_values.items()}
    #         for i in range(num_samples)
    #     ]

    #     grad = []
    #     # iterate over gradient vectors and bind the correct parameters
    #     for oper_i, param_i in zip(operator, param_bindings):
    #         # bind or re-bind remaining values and evaluate the gradient
    #         grad.append(oper_i.bind_parameters(param_i).eval())

    #     return np.asarray(grad)
