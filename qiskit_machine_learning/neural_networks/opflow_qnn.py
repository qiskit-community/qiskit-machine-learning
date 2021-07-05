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

"""An Opflow Quantum Neural Network that allows to use a parametrized opflow object as a
neural network."""
import logging
from typing import List, Optional, Union, Tuple

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
from qiskit.providers import BaseBackend, Backend
from qiskit.utils import QuantumInstance
from qiskit.utils.backend_utils import is_aer_provider

try:
    from sparse import SparseArray
except ImportError:

    class SparseArray:  # type: ignore
        """Empty SparseArray class
        Replacement if sparse.SparseArray is not present.
        """

        pass


from .neural_network import NeuralNetwork
from ..exceptions import QiskitMachineLearningError, QiskitError

logger = logging.getLogger(__name__)


class OpflowQNN(NeuralNetwork):
    """Opflow Quantum Neural Network."""

    def __init__(
        self,
        operator: OperatorBase,
        input_params: Optional[List[Parameter]] = None,
        weight_params: Optional[List[Parameter]] = None,
        exp_val: Optional[ExpectationBase] = None,
        gradient: Optional[Gradient] = None,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
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

        if isinstance(quantum_instance, (BaseBackend, Backend)):
            quantum_instance = QuantumInstance(quantum_instance)

        if quantum_instance:
            self._quantum_instance = quantum_instance
            self._circuit_sampler = CircuitSampler(
                self._quantum_instance,
                param_qobj=is_aer_provider(self._quantum_instance.backend),
                caching="all",
            )
        else:
            self._quantum_instance = None
            self._circuit_sampler = None

        self._operator = operator
        self._forward_operator = exp_val.convert(operator) if exp_val else operator
        self._gradient = gradient
        self._input_gradients = input_gradients
        self._construct_gradient_operator()

        output_shape = self._get_output_shape_from_op(operator)
        super().__init__(
            len(self._input_params),
            len(self._weight_params),
            sparse=False,
            output_shape=output_shape,
            input_gradients=input_gradients,
        )

    def _construct_gradient_operator(self):
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

    def _get_output_shape_from_op(self, op: OperatorBase) -> Tuple[int, ...]:
        """Determines the output shape of a given operator."""
        # TODO: the whole method should eventually be moved to opflow and rewritten in a better way.
        # if the operator is a composed one, then we only need to look at the first element of it.
        if isinstance(op, ComposedOp):
            return self._get_output_shape_from_op(op.oplist[0].primitive)
        # this "if" statement is on purpose, to prevent sub-classes.
        # pylint:disable=unidiomatic-typecheck
        if type(op) == ListOp:
            shapes = []
            for op_ in op.oplist:
                shape_ = self._get_output_shape_from_op(op_)
                shapes += [shape_]
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
        self._construct_gradient_operator()

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

        # check whether gradient circuit could be constructed
        if self._gradient_operator is None:
            return None, None

        # iterate over rows, each row is an element of a batch
        batch_size = input_data.shape[0]
        if self._input_gradients:
            num_params = self._num_inputs + self._num_weights
        else:
            num_params = self._num_weights

        grad_all = np.zeros((batch_size, *self._output_shape, num_params))

        for row in range(batch_size):
            # take i-th column as values for the i-th param in a batch
            param_values = {p: input_data[row, j] for j, p in enumerate(self._input_params)}
            param_values.update({p: weights[j] for j, p in enumerate(self._weight_params)})

            # evaluate gradient over all parameters
            if self._circuit_sampler:
                grad = self._circuit_sampler.convert(self._gradient_operator, param_values)
                # TODO: this should not be necessary and is a bug!
                grad = grad.bind_parameters(param_values)
                grad = np.real(grad.eval())
            else:
                grad = self._gradient_operator.bind_parameters(param_values)
                grad = np.real(grad.eval())
            grad_all[row, :] = grad.transpose()

        # split into and return input and weights gradients
        if self._input_gradients:
            input_grad = grad_all[:, :, : self._num_inputs].reshape(
                -1, *self._output_shape, self._num_inputs
            )

            weights_grad = grad_all[:, :, self._num_inputs :].reshape(
                -1, *self._output_shape, self._num_weights
            )
        else:
            input_grad = None
            weights_grad = grad_all.reshape(-1, *self._output_shape, self._num_weights)

        return input_grad, weights_grad
