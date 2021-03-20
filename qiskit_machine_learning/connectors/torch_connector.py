# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A connector to use Qiskit (Quantum) Neural Networks as PyTorch modules."""

from typing import Tuple, Any, Optional
import logging
import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError

from ..neural_networks import NeuralNetwork
from ..exceptions import QiskitMachineLearningError

logger = logging.getLogger(__name__)

try:
    from torch import Tensor, sparse_coo_tensor
    from torch.autograd import Function
    from torch.nn import Module, Parameter as TorchParam
except ImportError:
    class Function:  # type: ignore
        """ Empty Function class
            Replacement if torch.autograd.Function is not present.
        """
        pass

    class Tensor:  # type: ignore
        """ Empty Tensor class
            Replacement if torch.Tensor is not present.
        """
        pass

    class Module:  # type: ignore
        """ Empty Module class
            Replacement if torch.nn.Module is not present.
            Always fails to initialize
        """

        def __init__(self) -> None:
            raise MissingOptionalLibraryError(
                    libname='Pytorch',
                    name='TorchConnector',
                    pip_install="pip install 'qiskit-machine-learning[torch]'")


class TorchConnector(Module):
    """ Connects Qiskit (Quantum) Neural Network to PyTorch."""

    class _TorchNNFunction(Function):
        # pylint: disable=arguments-differ
        @staticmethod
        def forward(ctx: Any,  # type: ignore
                    input_data: Tensor,
                    weights: Tensor,
                    neural_network: NeuralNetwork,
                    sparse: bool) -> Tensor:
            """ Forward pass computation.
            Args:
                ctx: The context to be passed to the backward pass.
                input_data: The input data.
                weights: The weights.
                neural_network: The neural network to be connected.
                sparse: Indicates whether to use sparse output or not.

            Returns:
                The resulting value of the forward pass.

            Raises:
                QiskitMachineLearningError: Invalid input data.
            """

            # TODO: efficiently handle batches
            # validate input shape
            if input_data.shape[-1] != neural_network.num_inputs:
                raise QiskitMachineLearningError(
                    f'Invalid input dimension! Received {input_data.shape} and ' +
                    f'expected input compatible to {neural_network.num_inputs}')

            ctx.qnn = neural_network
            ctx.sparse = sparse
            ctx.save_for_backward(input_data, weights)
            result = neural_network.forward(input_data.numpy(), weights.numpy())
            if neural_network.sparse and sparse:
                result_tensor = sparse_coo_tensor(result.coords, result.data)
            else:
                result_tensor = Tensor(result)

            # if the input was not a batch, then remove the batch-dimension from the result,
            # since the neural network will always treat input as a batch and cast to a
            # single-element batch if no batch is given and PyTorch does not follow this
            #  convention.
            if len(input_data.shape) == 1:
                result_tensor = result_tensor[0]
            return result_tensor

        @staticmethod
        def backward(ctx: Any,  # type: ignore
                     grad_output: Tensor) -> Tuple:
            """ Backward pass computation.
            Args:
                ctx: context
                grad_output: previous gradient
            Raises:
                QiskitMachineLearningError: Invalid input data.
            Returns:
                gradients for the first two arguments and None for the others
            """

            # get context data
            input_data, weights = ctx.saved_tensors
            qnn = ctx.qnn

            # if sparse output is requested return None, since PyTorch does not support it yet.
            if qnn.sparse and ctx.sparse:
                return None, None, None, None

            # TODO: efficiently handle batches (for input and weights)
            # validate input shape
            if input_data.shape[-1] != qnn.num_inputs:
                raise QiskitMachineLearningError(
                    f'Invalid input dimension! Received {input_data.shape} and ' +
                    f' expected input compatible to {qnn.num_inputs}')

            # evaluate QNN gradient
            input_grad, weights_grad = qnn.backward(input_data.numpy(), weights.numpy())
            if input_grad is not None:
                if np.prod(input_grad.shape) == 0:
                    input_grad = None
                elif qnn.sparse:
                    # cast to dense here, since PyTorch does not support sparse output yet.
                    # this should only happen if the network returns sparse output but the
                    # connector is configured to return dense output.
                    input_grad = sparse_coo_tensor(input_grad.coords, input_grad.data)
                    input_grad = input_grad.to_dense()
                    if len(input_grad.shape) == 1:  # TODO: can be removed after batching
                        input_grad = input_grad.reshape(1, len(input_grad))

                    input_grad = grad_output.float() @ input_grad
                else:
                    if len(input_grad.shape) == 1:  # TODO: can be removed after batching is done
                        input_grad = input_grad.reshape(1, len(input_grad))
                    input_grad = grad_output.float() @ Tensor(input_grad)

            if weights_grad is not None:
                if np.prod(weights_grad.shape) == 0:
                    weights_grad = None
                elif qnn.sparse:
                    weights_grad = sparse_coo_tensor(weights_grad.coords, weights_grad.data)
                    if len(weights_grad.shape) == 1:  # TODO: can be removed after batching
                        weights_grad = weights_grad.reshape(1, len(weights_grad))
                    weights_grad = grad_output.float() @ weights_grad
                else:
                    if len(weights_grad.shape) == 1:  # TODO: can be removed after batching is done
                        weights_grad = weights_grad.reshape(1, len(weights_grad))
                    weights_grad = grad_output.float() @ Tensor(weights_grad)

            # return gradients for the first two arguments and None for the others (ie. qnn, sparse)
            return input_grad, weights_grad, None, None

    def __init__(self, neural_network: NeuralNetwork, sparse: Optional[bool] = None):
        """Initializes the TorchConnector.

        Args:
            neural_network: The neural network to be connected to PyTorch.
            sparse: Whether this connector should return sparse output or not. If sparse is set
                to None, then the setting from the given neural network is used. Note that sparse
                output is only returned if the underlying neural network also returns sparse output,
                otherwise it will be dense independent of the setting. Also note that PyTorch
                currently does not support sparse backpropagation, i.e., if sparse is set to True,
                the backward pass of this module will return None.
        """
        super().__init__()
        self._neural_network = neural_network
        self._sparse = sparse
        self._weights = TorchParam(Tensor(neural_network.num_weights))
        self._weights.data.uniform_(-1, 1)  # TODO: enable more reasonable initialization

    @property
    def neural_network(self):
        """ Returns the underlying neural network."""
        return self._neural_network

    @property
    def sparse(self):
        """ Returns whether this connector returns sparse output or not."""
        return self._sparse

    def forward(self, input_data: Optional[Tensor] = None) -> Tensor:
        """Forward pass.
        Args:
            input_data: data to be evaluated.
        Returns:
            Result of forward pass of this model.
        """
        input_ = input_data if input_data is not None else Tensor([])
        return TorchConnector._TorchNNFunction.apply(input_, self._weights,
                                                     self._neural_network, self._sparse)
