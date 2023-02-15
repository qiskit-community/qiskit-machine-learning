# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A connector to use Qiskit (Quantum) Neural Networks as PyTorch modules."""

from typing import Tuple, Any, Optional, cast, Union
import numpy as np

import qiskit_machine_learning.optionals as _optionals
from ..neural_networks import NeuralNetwork
from ..exceptions import QiskitMachineLearningError

if _optionals.HAS_TORCH:
    import torch
    from torch.autograd import Function
    from torch.nn import Module, Parameter as TorchParam
else:

    class Function:  # type: ignore
        """Empty Function class
        Replacement if torch.autograd.Function is not present.
        """

        pass

    class Tensor:  # type: ignore
        """Empty Tensor class
        Replacement if torch.Tensor is not present.
        """

        pass

    class Module:  # type: ignore
        """Empty Module class
        Replacement if torch.nn.Module is not present.
        """

        pass


@_optionals.HAS_TORCH.require_in_instance
class TorchConnector(Module):
    """Connects a Qiskit (Quantum) Neural Network to PyTorch."""

    class _TorchNNFunction(Function):
        # pylint: disable=arguments-differ
        @staticmethod
        def forward(  # type: ignore
            ctx: Any,
            input_data: torch.Tensor,
            weights: Tensor,
            neural_network: NeuralNetwork,
            sparse: bool,
        ) -> Tensor:
            """Forward pass computation.
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

            # validate input shape
            if input_data.shape[-1] != neural_network.num_inputs:
                raise QiskitMachineLearningError(
                    f"Invalid input dimension! Received {input_data.shape} and "
                    + f"expected input compatible to {neural_network.num_inputs}"
                )

            ctx.neural_network = neural_network
            ctx.sparse = sparse
            ctx.save_for_backward(input_data, weights)

            # Detach the tensors and move it to CPU as we need numpy array to compute gradients
            # of the quantum neural network. If the tensors are on CPU already this does nothing.
            # Some other tensors down below are also moved to CPU for computations.
            result = neural_network.forward(
                input_data.detach().cpu().numpy(), weights.detach().cpu().numpy()
            )
            if neural_network.sparse and sparse:
                _optionals.HAS_SPARSE.require_now("COO")
                # pylint: disable=import-error
                from sparse import SparseArray, COO

                result = cast(COO, cast(SparseArray, result).asformat("coo"))
                result_tensor = torch.sparse_coo_tensor(result.coords, result.data)
            elif neural_network.sparse and not sparse:
                # convert from a sparse tensor to dense
                result_tensor = torch.from_numpy(result.todense()).to(input_data.dtype)
            else:
                # result_tensor = Tensor(result)
                result_tensor = torch.from_numpy(result).to(input_data.dtype)

            # if the input was not a batch, then remove the batch-dimension from the result,
            # since the neural network will always treat input as a batch and cast to a
            # single-element batch if no batch is given and PyTorch does not follow this
            # convention.
            if len(input_data.shape) == 1:
                result_tensor = result_tensor[0]

            # place the resulting tensor back to the device where input data is stored
            result_tensor = result_tensor.to(input_data.device)

            return result_tensor

        @staticmethod
        def backward(ctx: Any, grad_output: Tensor) -> Tuple:  # type: ignore
            """Backward pass computation.
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
            neural_network = ctx.neural_network

            # if sparse output is requested return None, since PyTorch does not support it yet.
            if neural_network.sparse and ctx.sparse:
                return None, None, None, None

            # validate input shape
            if input_data.shape[-1] != neural_network.num_inputs:
                raise QiskitMachineLearningError(
                    f"Invalid input dimension! Received {input_data.shape} and "
                    + f" expected input compatible to {neural_network.num_inputs}"
                )

            # ensure same shape for single observations and batch mode
            if len(grad_output.shape) == 1:
                grad_output = grad_output.view(1, -1)

            # evaluate QNN gradient
            input_grad, weights_grad = neural_network.backward(
                input_data.detach().cpu().numpy(), weights.detach().cpu().numpy()
            )
            if input_grad is not None:
                if neural_network.sparse:
                    input_grad = torch.sparse_coo_tensor(input_grad.coords, input_grad.data)

                    # cast to dense here, since PyTorch does not support sparse output yet.
                    # this should only happen if the network returns sparse output but the
                    # connector is configured to return dense output.
                    input_grad = input_grad.to_dense()  # this should be eventually removed
                    input_grad = input_grad.to(grad_output.dtype)
                else:
                    input_grad = torch.from_numpy(input_grad).to(grad_output.dtype)

                # Takes gradients from previous layer in backward pass (i.e. later layer in forward
                # pass) j for each observation i in the batch. Multiplies this with the gradient
                # from this point on backwards with respect to each input k. Sums over all j
                # to get total gradient of output w.r.t. each input k and batch index i.
                # This operation should preserve the batch dimension to be able to do back-prop in
                # a batched manner.
                input_grad = torch.einsum("ij,ijk->ik", grad_output.detach().cpu(), input_grad)

                # place the resulting tensor to the device where they were stored
                input_grad = input_grad.to(input_data.device)

            if weights_grad is not None:
                if neural_network.sparse:
                    weights_grad = torch.sparse_coo_tensor(weights_grad.coords, weights_grad.data)

                    # cast to dense here, since PyTorch does not support sparse output yet.
                    # this should only happen if the network returns sparse output but the
                    # connector is configured to return dense output.
                    weights_grad = weights_grad.to_dense()  # this should be eventually removed
                    weights_grad = weights_grad.to(grad_output.dtype)
                else:
                    weights_grad = torch.from_numpy(weights_grad).to(grad_output.dtype)

                # Takes gradients from previous layer in backward pass (i.e. later layer in forward
                # pass) j for each observation i in the batch. Multiplies this with the gradient
                # from this point on backwards with respect to each parameter k. Sums over all i and
                # j to get total gradient of output w.r.t. each parameter k.
                # The weights' dimension is independent of the batch size.
                weights_grad = torch.einsum("ij,ijk->k", grad_output.detach().cpu(), weights_grad)

                # place the resulting tensor to the device where they were stored
                weights_grad = weights_grad.to(weights.device)

            # return gradients for the first two arguments and None for the others (i.e. qnn/sparse)
            return input_grad, weights_grad, None, None

    def __init__(
        self,
        neural_network: NeuralNetwork,
        initial_weights: Optional[Union[np.ndarray, Tensor]] = None,
        sparse: Optional[bool] = None,
    ):
        """
        Args:
            neural_network: The neural network to be connected to PyTorch. Remember
                    that ``input_gradients``  must be set to ``True`` in the neural network
                    initialization before passing it to the ``TorchConnector`` for the gradient
                    computations to work properly during training.
            initial_weights: The initial weights to start training the network. If this is None,
                the initial weights are chosen uniformly at random from [-1, 1].
            sparse: Whether this connector should return sparse output or not. If sparse is set
                to None, then the setting from the given neural network is used. Note that sparse
                output is only returned if the underlying neural network also returns sparse output,
                otherwise it will be dense independent of the setting. Also note that PyTorch
                currently does not support sparse back propagation, i.e., if sparse is set to True,
                the backward pass of this module will return None.
        """
        super().__init__()
        self._neural_network = neural_network
        self._sparse = sparse

        weight_param = TorchParam(Tensor(neural_network.num_weights))
        # Register param. in graph following PyTorch naming convention
        self.register_parameter("weight", weight_param)
        # If `weight_param` is assigned to `self._weights` after registration,
        # it will not be re-registered, and we can keep the private var. name
        # "_weights" for compatibility. The alternative, doing:
        # `self._weights = TorchParam(Tensor(neural_network.num_weights))`
        # would register the parameter with the name "_weights".
        self._weights = weight_param

        if initial_weights is None:
            self._weights.data.uniform_(-1, 1)
        else:
            self._weights.data = Tensor(initial_weights)

    @property
    def neural_network(self) -> NeuralNetwork:
        """Returns the underlying neural network."""
        return self._neural_network

    @property
    def weight(self) -> Tensor:
        """Returns the weights of the underlying network."""
        return self._weights

    @property
    def sparse(self) -> Optional[bool]:
        """Returns whether this connector returns sparse output or not."""
        return self._sparse

    def forward(self, input_data: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Args:
            input_data: data to be evaluated.

        Returns:
            Result of forward pass of this model.
        """
        input_ = input_data if input_data is not None else torch.empty(0)
        return TorchConnector._TorchNNFunction.apply(
            input_, self._weights, self._neural_network, self._sparse
        )
