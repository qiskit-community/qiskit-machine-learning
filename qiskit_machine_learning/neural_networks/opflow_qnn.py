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

from copy import deepcopy
from typing import List, Optional, Union, Tuple

import numpy as np
from sparse import SparseArray

from qiskit.circuit import Parameter
from qiskit.opflow import Gradient, CircuitSampler, ListOp, OperatorBase, ExpectationBase
from qiskit.providers import BaseBackend, Backend
from qiskit.utils import QuantumInstance
from qiskit.utils.backend_utils import is_aer_provider

from .neural_network import NeuralNetwork
from .. import QiskitMachineLearningError


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
        #  TODO: most of these should be private properties
        self.operator = operator
        self.input_params = list(input_params or [])
        self.weight_params = list(weight_params or [])
        self.exp_val = exp_val  # TODO: currently not used by Gradient!
        self.gradient = gradient or Gradient()

        if isinstance(quantum_instance, (BaseBackend, Backend)):
            quantum_instance = QuantumInstance(quantum_instance)

        if quantum_instance:
            self.quantum_instance = quantum_instance
            self.circuit_sampler = CircuitSampler(
                self.quantum_instance,
                param_qobj=is_aer_provider(self.quantum_instance.backend)
            )
            # TODO: replace by extended caching in circuit sampler after merged: "caching='all'"
            self.gradient_sampler = deepcopy(self.circuit_sampler)
        else:
            self.quantum_instance = None
            self.circuit_sampler = None
            self.gradient_sampler = None

        self.forward_operator = self.exp_val.convert(operator) if exp_val else operator
        self.gradient_operator = self.gradient.convert(operator,
                                                       self.input_params + self.weight_params)
        output_shape = self._get_output_shape_from_op(operator)
        super().__init__(len(self.input_params), len(self.weight_params), True, output_shape)

    def _get_output_shape_from_op(self, op: OperatorBase) -> Tuple[int, ...]:
        """Determines the output shape of a given operator."""
        # TODO: should eventually be moved to opflow
        if isinstance(op, ListOp):
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

    def _forward(self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
                 ) -> Union[np.ndarray, SparseArray]:
        # combine parameter dictionary
        param_values = {p: input_data[i] for i, p in enumerate(self.input_params)}
        param_values.update({p: weights[i] for i, p in enumerate(self.weight_params)})

        # evaluate operator
        if self.circuit_sampler:
            op = self.circuit_sampler.convert(self.forward_operator, param_values)
            result = np.real(op.eval())
        else:
            op = self.forward_operator.bind_parameters(param_values)
            result = np.real(op.eval())
        result = np.array(result)
        return result.reshape(1, *self.output_shape)

    def _backward(self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
                  ) -> Tuple[Optional[Union[np.ndarray, SparseArray]],
                             Optional[Union[np.ndarray, SparseArray]]]:
        # combine parameter dictionary
        param_values = {p: input_data[i] for i, p in enumerate(self.input_params)}
        param_values.update({p: weights[i] for i, p in enumerate(self.weight_params)})

        # evaluate gradient over all parameters
        if self.gradient_sampler:
            grad = self.gradient_sampler.convert(self.gradient_operator, param_values)
            # TODO: this should not be necessary and is a bug!
            grad = grad.bind_parameters(param_values)
            grad = np.real(grad.eval())
        else:
            grad = self.gradient_operator.bind_parameters(param_values)
            grad = np.real(grad.eval())

        # split into and return input and weights gradients
        input_grad = np.array(grad[:len(input_data)]).reshape(
            1, *self.output_shape, self.num_inputs)

        weights_grad = np.array(grad[len(input_data):]).reshape(
            1, *self.output_shape, self.num_weights)

        return input_grad, weights_grad
