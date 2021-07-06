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

from typing import Tuple, Any, Optional, cast, Union
import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError

try:
    from sparse import SparseArray, COO

    _HAS_SPARSE = True
except ImportError:
    _HAS_SPARSE = False

from ..neural_networks import NeuralNetwork
from ..exceptions import QiskitMachineLearningError
from ..deprecation import deprecate_property

try:
    from torch import Tensor, sparse_coo_tensor, einsum
    from torch.autograd import Function
    from torch.nn import Module, Parameter as TorchParam
except ImportError:

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
        Always fails to initialize
        """

        def __init__(self) -> None:
            raise MissingOptionalLibraryError(
                libname="Pytorch",
                name="TorchConnector",
                pip_install="pip install 'qiskit-machine-learning[torch]'",
            )


class TorchConnector(Module):
    """Connects a Qiskit (Quantum) Neural Network to PyTorch."""

    class _TorchNNFunction(Function):
        # pylint: disable=arguments-differ
        @staticmethod
        def forward(  # type: ignore
            ctx: Any,
            input_data: Tensor,
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
                MissingOptionalLibraryError: sparse not installed.
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
            result = neural_network.forward(input_data.numpy(), weights.numpy())
            if neural_network.sparse and sparse:
                if not _HAS_SPARSE:
                    raise MissingOptionalLibraryError(
                        libname="sparse",
                        name="COO",
                        pip_install="pip install 'qiskit-machine-learning[sparse]'",
                    )
                result = cast(COO, cast(SparseArray, result).asformat("coo"))
                result_tensor = sparse_coo_tensor(result.coords, result.data)
            else:
                result_tensor = Tensor(result)

            # if the input was not a batch, then remove the batch-dimension from the result,
            # since the neural network will always treat input as a batch and cast to a
            # single-element batch if no batch is given and PyTorch does not follow this
            # convention.
            if len(input_data.shape) == 1:
                result_tensor = result_tensor[0]

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
            input_grad, weights_grad = neural_network.backward(input_data.numpy(), weights.numpy())
            if input_grad is not None:
                if np.prod(input_grad.shape) == 0:
                    input_grad = None
                elif neural_network.sparse:
                    input_grad = sparse_coo_tensor(input_grad.coords, input_grad.data)

                    # cast to dense here, since PyTorch does not support sparse output yet.
                    # this should only happen if the network returns sparse output but the
                    # connector is configured to return dense output.
                    input_grad = input_grad.to_dense()  # this should be eventually removed
                    input_grad = input_grad.to(grad_output.dtype)
                else:
                    input_grad = Tensor(input_grad).to(grad_output.dtype)

                # Takes gradients from previous layer in backward pass (i.e. later layer in forward
                # pass) j for each observation i in the batch. Multiplies this with the gradient
                # from this point on backwards with respect to each input k. Sums over all j
                # to get total gradient of output w.r.t. each input k and batch index i.
                # This operation should preserve the batch dimension to be able to do back-prop in
                # a batched manner.
                input_grad = einsum("ij,ijk->ik", grad_output, input_grad)

            if weights_grad is not None:
                if np.prod(weights_grad.shape) == 0:
                    weights_grad = None
                elif neural_network.sparse:
                    weights_grad = sparse_coo_tensor(weights_grad.coords, weights_grad.data)

                    # cast to dense here, since PyTorch does not support sparse output yet.
                    # this should only happen if the network returns sparse output but the
                    # connector is configured to return dense output.
                    weights_grad = weights_grad.to_dense()  # this should be eventually removed
                    weights_grad = weights_grad.to(grad_output.dtype)
                else:
                    weights_grad = Tensor(weights_grad).to(grad_output.dtype)

                # Takes gradients from previous layer in backward pass (i.e. later layer in forward
                # pass) j for each observation i in the batch. Multiplies this with the gradient
                # from this point on backwards with respect to each parameter k. Sums over all i and
                # j to get total gradient of output w.r.t. each parameter k.
                # The weights' dimension is independent of the batch size.
                weights_grad = einsum("ij,ijk->k", grad_output, weights_grad)

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
        self._weights = TorchParam(Tensor(neural_network.num_weights))
        if initial_weights is None:
            self._weights.data.uniform_(-1, 1)
        else:
            self._weights.data = Tensor(initial_weights)

    @property
    def neural_network(self) -> NeuralNetwork:
        """Returns the underlying neural network."""
        return self._neural_network

    # Bug in mypy, if property decorator is used with another one
    # https://github.com/python/mypy/issues/1362

    @property  # type: ignore
    @deprecate_property("0.2.0", new_name="weight")
    def weights(self) -> Tensor:
        """
        .. deprecated:: 0.2.0
           Use :meth:`weight` instead.

        Returns the weights of the underlying network.
        """
        return self.weight

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
        input_ = input_data if input_data is not None else Tensor([])
        return TorchConnector._TorchNNFunction.apply(
            input_, self._weights, self._neural_network, self._sparse
        )
