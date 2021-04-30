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
from qiskit.opflow import Gradient, CircuitSampler, ListOp, OperatorBase, ExpectationBase, \
    OpflowError
from qiskit.providers import BaseBackend, Backend
from qiskit.utils import QuantumInstance
from qiskit.utils.backend_utils import is_aer_provider
from sparse import SparseArray

from .neural_network import NeuralNetwork
from ..exceptions import QiskitMachineLearningError, QiskitError

logger = logging.getLogger(__name__)


class OpflowQNN(NeuralNetwork):
    """Opflow Quantum Neural Network."""

    def __init__(self, operator: OperatorBase,
                 input_params: Optional[List[Parameter]] = None,
                 weight_params: Optional[List[Parameter]] = None,
                 exp_val: Optional[ExpectationBase] = None,
                 gradient: Optional[Gradient] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None):
        """Initializes the Opflow Quantum Neural Network.

        Args:
            operator: The parametrized operator that represents the neural network.
            input_params: The operator parameters that correspond to the input of the network.
            weight_params: The operator parameters that correspond to the trainable weights.
            exp_val: The Expected Value converter to be used for the operator.
            gradient: The Gradient converter to be used for the operator's backward pass.
            quantum_instance: The quantum instance to evaluate the network.
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
                caching="all"
            )
        else:
            self._quantum_instance = None
            self._circuit_sampler = None

        self._operator = operator
        self._forward_operator = exp_val.convert(operator) if exp_val else operator
        self._gradient_operator: OperatorBase = None
        try:
            gradient = gradient or Gradient()
            self._gradient_operator = gradient.convert(operator,
                                                       self._input_params + self._weight_params)
        except (ValueError, TypeError, OpflowError, QiskitError):
            logger.warning('Cannot compute gradient operator! Continuing without gradients!')

        output_shape = self._get_output_shape_from_op(operator)
        super().__init__(len(self._input_params), len(self._weight_params),
                         sparse=False, output_shape=output_shape)

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Returns the quantum instance to evaluate the circuit."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance) -> None:
        """Sets the quantum instance to evaluate the circuit."""
        self._quantum_instance = quantum_instance

    def _get_output_shape_from_op(self, op: OperatorBase) -> Tuple[int, ...]:
        """Determines the output shape of a given operator."""
        # TODO: should eventually be moved to opflow
        # this "if" statement is on purpose, to prevent sub-classes.
        # pylint:disable=unidiomatic-typecheck
        if type(op) == ListOp:
            shapes = []
            for op_ in op.oplist:
                shape_ = self._get_output_shape_from_op(op_)
                shapes += [shape_]
            if not np.all([shape == shapes[0] for shape in shapes]):
                raise QiskitMachineLearningError(
                    'Only supports ListOps with children that return the same shape.')
            if shapes[0] == (1,):
                out = op.combo_fn(np.zeros((len(op.oplist))))
            else:
                out = op.combo_fn(np.zeros((len(op.oplist), *shapes[0])))
            return out.shape
        else:
            return (1,)

    @property
    def operator(self):
        """ Returns the underlying operator of this QNN."""
        return self._operator

    def _forward(self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
                 ) -> Union[np.ndarray, SparseArray]:
        # combine parameter dictionary
        # take i-th column as values for the i-th param in a batch
        param_values = {p: input_data[:, i].tolist() for i, p in enumerate(self._input_params)}
        param_values.update({p: [weights[i]] * input_data.shape[0]
                             for i, p in enumerate(self._weight_params)})

        # evaluate operator
        if self._circuit_sampler:
            op = self._circuit_sampler.convert(self._forward_operator, param_values)
            result = np.real(op.eval())
        else:
            op = self._forward_operator.bind_parameters(param_values)
            result = np.real(op.eval())
        result = np.array(result)
        return result.reshape(-1, *self.output_shape)

    def _backward(self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
                  ) -> Tuple[Optional[Union[np.ndarray, SparseArray]],
                             Optional[Union[np.ndarray, SparseArray]]]:

        # check whether gradient circuit could be constructed
        if self._gradient_operator is None:
            return None, None

        # iterate over rows, each row is an element of a batch
        batch_size = input_data.shape[0]
        grad_all = np.zeros((batch_size, *self.output_shape, self.num_inputs + self.num_weights))
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
        input_grad = np.array(grad_all[:batch_size, :, :self.num_inputs])\
            .reshape(-1, *self.output_shape, self.num_inputs)

        weights_grad = np.array(grad_all[:batch_size, :, self.num_inputs:])\
            .reshape(-1, *self.output_shape, self.num_weights)

        return input_grad, weights_grad
