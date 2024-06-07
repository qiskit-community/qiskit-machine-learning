# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A connector to use Qiskit (Quantum) Neural Networks as PyTorch modules."""
from __future__ import annotations

from typing import Tuple, Any, cast

from string import ascii_lowercase
import numpy as np

import qiskit_machine_learning.optionals as _optionals
from ..exceptions import QiskitMachineLearningError
from ..neural_networks import NeuralNetwork

if _optionals.HAS_TORCH:
    import torch

    # Imports for inheritance and type hints
    from torch import Tensor
    from torch.autograd import Function
    from torch.nn import Module
else:
    import types

    # Dummy definition for torch module to prevent lint error about
    # possible use of torch before assignment
    torch = types.ModuleType("torch")

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


CHAR_LIMIT = 26


def _get_einsum_signature(n_dimensions: int, for_weights: bool = False) -> str:
    """
    Generate an Einstein summation signature for a given number of dimensions and return type.

    Args:
        n_dimensions (int): The number of dimensions for the summation.
        for_weights (bool): If True, the return signature includes only the
            last index as the output. If False, the return signature includes
            all input indices except the last one. Defaults to False.


    Returns:
        str: The Einstein summation signature.

    Raises:
        RuntimeError: If the number of dimensions exceeds the character limit.
    """
    if n_dimensions > CHAR_LIMIT - 1:
        raise RuntimeError(
            f"Cannot define an Einstein summation with more than {CHAR_LIMIT - 1:d} dimensions, "
            f"got {n_dimensions:d}."
        )

    trace = ascii_lowercase[:n_dimensions]

    if for_weights:
        return f"{trace[:-1]},{trace:s}->{trace[-1]}"

    return f"{trace[:-1]},{trace:s}->{trace[0] + trace[2:]}"


@_optionals.HAS_TORCH.require_in_instance
class TorchConnector(Module):
    """Connects a Qiskit (Quantum) Neural Network to PyTorch.
    The dataset dimensionality is limited to 25. Datasets of 26 dimensions or higher
    raise a `RuntimeError`.

    """

    # pylint: disable=abstract-method
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
                RuntimeError: if connector is configured as sparse and the network is not sparse.
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
            if ctx.sparse:
                if neural_network.sparse:
                    _optionals.HAS_SPARSE.require_now("SparseArray")
                    # pylint: disable=import-error
                    from sparse import SparseArray, COO

                    # todo: replace output type from DOK to COO?
                    result = cast(COO, cast(SparseArray, result).asformat("coo"))
                    result_tensor = torch.sparse_coo_tensor(result.coords, result.data)
                else:
                    raise RuntimeError(
                        "TorchConnector configured as sparse, the network must be sparse as well"
                    )
            else:
                # connector is dense
                if neural_network.sparse:
                    # convert to dense
                    _optionals.HAS_SPARSE.require_now("SparseArray")
                    from sparse import SparseArray

                    # cast is required by mypy
                    result = cast(SparseArray, result).todense()
                result_tensor = torch.as_tensor(result, dtype=torch.float)

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
                RuntimeError: if connector is configured as sparse and the network is not sparse.

            Returns:
                gradients for the first two arguments and None for the others
            """

            # get context data
            input_data, weights = ctx.saved_tensors
            neural_network = ctx.neural_network

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
                if ctx.sparse:
                    if neural_network.sparse:
                        _optionals.HAS_SPARSE.require_now("Sparse")
                        import sparse
                        from sparse import COO

                        grad_output = grad_output.detach().cpu()
                        grad_coo = COO(grad_output.indices(), grad_output.values())

                        # Takes gradients from previous layer in backward pass (i.e. later layer in
                        # forward pass) j for each observation i in the batch. Multiplies this with
                        # the gradient from this point on backwards with respect to each input k.
                        # Sums over all j to get total gradient of output w.r.t. each input k and
                        # batch index i. This operation should preserve the batch dimension to be
                        # able to do back-prop in a batched manner.
                        # Pytorch does not support sparse einsum, so we rely on Sparse.
                        # pylint: disable=no-member
                        n_dimension = max(grad_coo.ndim, input_grad.ndim)
                        signature = _get_einsum_signature(n_dimension)
                        input_grad = sparse.einsum(signature, grad_coo, input_grad)

                        # return sparse gradients
                        input_grad = torch.sparse_coo_tensor(input_grad.coords, input_grad.data)
                    else:
                        # this exception should never happen
                        raise RuntimeError(
                            "TorchConnector configured as sparse, "
                            "the network must be sparse as well"
                        )
                else:
                    # connector is dense
                    if neural_network.sparse:
                        # convert to dense
                        input_grad = input_grad.todense()
                    input_grad = torch.as_tensor(input_grad, dtype=torch.float)

                    # same as above
                    n_dimension = max(grad_output.detach().cpu().ndim, input_grad.ndim)
                    signature = _get_einsum_signature(n_dimension)
                    input_grad = torch.einsum(signature, grad_output.detach().cpu(), input_grad)

                # place the resulting tensor to the device where they were stored
                input_grad = input_grad.to(input_data.device)

            if weights_grad is not None:
                if ctx.sparse:
                    if neural_network.sparse:
                        import sparse
                        from sparse import COO

                        grad_output = grad_output.detach().cpu()
                        grad_coo = COO(grad_output.indices(), grad_output.values())

                        # Takes gradients from previous layer in backward pass (i.e. later layer in
                        # forward pass) j for each observation i in the batch. Multiplies this with
                        # the gradient from this point on backwards with respect to each
                        # parameter k. Sums over all i and j to get total gradient of output
                        # w.r.t. each parameter k. The weights' dimension is independent of the
                        # batch size.
                        # pylint: disable=no-member
                        n_dimension = max(grad_coo.ndim, weights_grad.ndim)
                        signature = _get_einsum_signature(n_dimension, for_weights=True)
                        weights_grad = sparse.einsum(signature, grad_coo, weights_grad)

                        # return sparse gradients
                        weights_grad = torch.sparse_coo_tensor(
                            weights_grad.coords, weights_grad.data
                        )
                    else:
                        # this exception should never happen
                        raise RuntimeError(
                            "TorchConnector configured as sparse, "
                            "the network must be sparse as well"
                        )
                else:
                    if neural_network.sparse:
                        # convert to dense
                        weights_grad = weights_grad.todense()
                    weights_grad = torch.as_tensor(weights_grad, dtype=torch.float)
                    # same as above
                    n_dimension = max(grad_output.detach().cpu().ndim, weights_grad.ndim)
                    signature = _get_einsum_signature(n_dimension, for_weights=True)
                    weights_grad = torch.einsum(signature, grad_output.detach().cpu(), weights_grad)

                # place the resulting tensor to the device where they were stored
                weights_grad = weights_grad.to(weights.device)

            # return gradients for the first two arguments and None for the others (i.e. qnn/sparse)
            return input_grad, weights_grad, None, None

    def __init__(
        self,
        neural_network: NeuralNetwork,
        initial_weights: np.ndarray | Tensor | None = None,
        sparse: bool | None = None,
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
                otherwise an error will be raised.

        Raises:
            QiskitMachineLearningError: If the connector is configured as sparse and the underlying
                network is not sparse.
        """
        super().__init__()
        self._neural_network = neural_network
        if sparse is None:
            sparse = self._neural_network.sparse

        self._sparse = sparse

        if self._sparse and not self._neural_network.sparse:
            # connector is sparse while the underlying neural network is not
            raise QiskitMachineLearningError(
                "TorchConnector configured as sparse, the network must be sparse as well"
            )

        weight_param = torch.nn.Parameter(torch.zeros(neural_network.num_weights))
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
            self._weights.data = torch.tensor(initial_weights, dtype=torch.float)

    @property
    def neural_network(self) -> NeuralNetwork:
        """Returns the underlying neural network."""
        return self._neural_network

    @property
    def weight(self) -> Tensor:
        """Returns the weights of the underlying network."""
        return self._weights

    @property
    def sparse(self) -> bool | None:
        """Returns whether this connector returns sparse output or not."""
        return self._sparse

    def forward(self, input_data: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            input_data: data to be evaluated.

        Returns:
            Result of forward pass of this model.
        """
        input_ = input_data if input_data is not None else torch.zeros(0)
        return TorchConnector._TorchNNFunction.apply(
            input_, self._weights, self._neural_network, self._sparse
        )
