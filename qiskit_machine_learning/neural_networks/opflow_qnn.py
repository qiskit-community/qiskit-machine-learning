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
from qiskit.circuit import Parameter
from qiskit.opflow import (
    Gradient,
    CircuitSampler,
    ListOp,
    OperatorBase,
    ExpectationBase,
    OpflowError,
    ComposedOp,
)
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance, deprecate_function
from qiskit.utils.backend_utils import is_aer_provider
import qiskit_machine_learning.optionals as _optionals
from .neural_network import NeuralNetwork
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


class OpflowQNN(NeuralNetwork):
    """Pending deprecation: Opflow Quantum Neural Network."""

    @deprecate_function(
        "The OpflowQNN class has been superseded by the "
        "qiskit_machine_learning.neural_networks.EstimatorQNN "
        "This class will be deprecated in a future release and subsequently "
        "removed after that.",
        stacklevel=3,
        category=PendingDeprecationWarning,
    )
    def __init__(
        self,
        operator: OperatorBase,
        input_params: Optional[List[Parameter]] = None,
        weight_params: Optional[List[Parameter]] = None,
        exp_val: Optional[ExpectationBase] = None,
        gradient: Optional[Gradient] = None,
        quantum_instance: Optional[Union[QuantumInstance, Backend]] = None,
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
        self._input_params = list(input_params) or []
        self._weight_params = list(weight_params) or []
        self._set_quantum_instance(quantum_instance)
        self._operator = operator
        self._forward_operator = exp_val.convert(operator) if exp_val else operator
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

    def _construct_gradient_operator(self):
        if self._gradient_operator_constructed:
            return

        self._gradient_operator: OperatorBase = None
        try:
            gradient = self._gradient or Gradient()
            if self._input_gradients:
                params = self._input_params + self._weight_params
            else:
                params = self._weight_params

            self._gradient_operator = gradient.convert(self._operator, params)
        except (ValueError, TypeError, OpflowError, QiskitError):
            logger.warning("Cannot compute gradient operator! Continuing without gradients!")

        self._gradient_operator_constructed = True

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

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Returns the quantum instance to evaluate the operator."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Optional[Union[QuantumInstance, Backend]]) -> None:
        """Sets the quantum instance to evaluate the operator."""
        self._set_quantum_instance(quantum_instance)

    def _set_quantum_instance(
        self, quantum_instance: Optional[Union[QuantumInstance, Backend]]
    ) -> None:
        """
        Internal method to set a quantum instance and compute/initialize a sampler.

        Args:
            quantum_instance: A quantum instance to set.

        Returns:
            None.
        """

        if isinstance(quantum_instance, Backend):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance

        if quantum_instance is not None:
            self._circuit_sampler = CircuitSampler(
                self._quantum_instance,
                param_qobj=is_aer_provider(self._quantum_instance.backend),
                caching="all",
            )
        else:
            self._circuit_sampler = None

    def _forward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Union[np.ndarray, SparseArray]:
        # combine parameter dictionary
        # take i-th column as values for the i-th param in a batch
        param_values = {p: input_data[:, i].tolist() for i, p in enumerate(self._input_params)}
        param_values.update(
            {p: [weights[i]] * input_data.shape[0] for i, p in enumerate(self._weight_params)}
        )

        # evaluate operator
        if self._circuit_sampler:
            op = self._circuit_sampler.convert(self._forward_operator, param_values)
            result = np.real(op.eval())
        else:
            op = self._forward_operator.bind_parameters(param_values)
            result = np.real(op.eval())
        result = np.array(result)
        return result.reshape(-1, *self._output_shape)

    def _backward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Tuple[Optional[Union[np.ndarray, SparseArray]], Optional[Union[np.ndarray, SparseArray]],]:

        self._construct_gradient_operator()

        # check whether gradient circuit could be constructed
        if self._gradient_operator is None:
            return None, None

        num_samples = input_data.shape[0]
        if self._input_gradients:
            num_params = self._num_inputs + self._num_weights
        else:
            num_params = self._num_weights

        param_values = {
            input_param: input_data[:, j] for j, input_param in enumerate(self._input_params)
        }
        param_values.update(
            {
                weight_param: np.full(num_samples, weights[j])
                for j, weight_param in enumerate(self._weight_params)
            }
        )

        if self._circuit_sampler:
            converted_op = self._circuit_sampler.convert(self._gradient_operator, param_values)
            # if statement is a workaround for https://github.com/Qiskit/qiskit-terra/issues/7608
            if len(converted_op.parameters) > 0:
                # rebind the leftover parameters and evaluate the gradient
                grad = self._evaluate_operator(converted_op, num_samples, param_values)
            else:
                # all parameters are bound by CircuitSampler, so we evaluate the operator directly
                grad = np.asarray(converted_op.eval())
        else:
            # we evaluate gradient operator for each sample separately, so we create a list of operators.
            grad = self._evaluate_operator(
                [self._gradient_operator] * num_samples, num_samples, param_values
            )

        grad = np.real(grad)

        # this is another workaround to fix output shape of the invocation result of CircuitSampler
        if self._output_shape == (1,):
            # at least 3 dimensions: batch, output, num_parameters, but in this case we don't have
            # output dimension, so we add a dimension that corresponds to the output
            grad = grad.reshape((num_samples, 1, num_params))
        else:
            # swap last axis that corresponds to parameters and axes correspond to the output shape
            last_axis = len(grad.shape) - 1
            grad = grad.transpose([0, last_axis, *(range(1, last_axis))])

        # split into and return input and weights gradients
        if self._input_gradients:
            input_grad = grad[:, :, : self._num_inputs].reshape(
                -1, *self._output_shape, self._num_inputs
            )

            weights_grad = grad[:, :, self._num_inputs :].reshape(
                -1, *self._output_shape, self._num_weights
            )
        else:
            input_grad = None
            weights_grad = grad.reshape(-1, *self._output_shape, self._num_weights)

        return input_grad, weights_grad

    def _evaluate_operator(
        self,
        operator: Union[OperatorBase, List[OperatorBase]],
        num_samples: int,
        param_values: Dict[Parameter, np.ndarray],
    ) -> np.ndarray:
        """
        Evaluates an operator or a list of operators for the samples in the dataset. If an operator
        is passed then it is considered as an iterable that has `num_samples` elements. Usually such
        operators are obtained as an output from `CircuitSampler`. If a list of operators is passed
        then each operator in this list is evaluated with a set of values/parameters corresponding
        to the sample index in the `param_values` as the operator in the list.

        Args:
            operator: operator or list of operators to evaluate.
            num_samples: a total number of samples
            param_values: parameter values to use for operator evaluation.

        Returns:
            the result of operator evaluation as an array.
        """
        # create an list of parameter bindings, each element corresponds to a sample in the dataset
        param_bindings = [
            {param: param_values[i] for param, param_values in param_values.items()}
            for i in range(num_samples)
        ]

        grad = []
        # iterate over gradient vectors and bind the correct parameters
        for oper_i, param_i in zip(operator, param_bindings):
            # bind or re-bind remaining values and evaluate the gradient
            grad.append(oper_i.bind_parameters(param_i).eval())

        return np.asarray(grad)
