# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
PyTorch Discriminator Neural Network
"""

from typing import Dict, Any, Iterable, Optional, Sequence, cast
import os
import numpy as np
from qiskit.utils import QuantumInstance
import qiskit_machine_learning.optionals as _optionals
from ....deprecation import deprecate_function
from .discriminative_network import DiscriminativeNetwork


@_optionals.HAS_TORCH.require_in_instance
class PyTorchDiscriminator(DiscriminativeNetwork):
    """
    Discriminator based on PyTorch
    """

    @deprecate_function(
        "0.5.0",
        additional_msg="with no direct replacement for it. "
        "Instead, please refer to the new QGAN tutorial",
        stack_level=3,
    )
    def __init__(self, n_features: int = 1, n_out: int = 1) -> None:
        """
        Args:
            n_features: Dimension of input data vector.
            n_out: Dimension of the discriminator's output vector.
        """
        super().__init__()
        # pylint: disable=import-outside-toplevel
        from torch import optim

        self._n_features = n_features
        self._n_out = n_out
        # discriminator_net: torch.nn.Module or None, Discriminator network.
        # pylint: disable=import-outside-toplevel
        from ._pytorch_discriminator_net import DiscriminatorNet

        self._discriminator = DiscriminatorNet(self._n_features, self._n_out)
        # optimizer: torch.optim.Optimizer or None, Optimizer initialized w.r.t
        # discriminator network parameters.
        self._optimizer = optim.Adam(self._discriminator.parameters(), lr=1e-5, amsgrad=True)

        self._ret = {}  # type: Dict[str, Any]

    def set_seed(self, seed: int):
        """
        Set seed.

        Args:
            seed: seed
        """
        import torch

        torch.manual_seed(seed)

    def save_model(self, snapshot_dir: str):
        """
        Save discriminator model

        Args:
            snapshot_dir:  directory path for saving the model
        """
        import torch

        torch.save(self._discriminator, os.path.join(snapshot_dir, "discriminator.pt"))

    def load_model(self, load_dir: str):
        """
        Load discriminator model

        Args:
            load_dir: file with stored PyTorch discriminator model to be loaded
        """
        import torch

        self._discriminator = torch.load(load_dir)

    @property
    def discriminator_net(self):
        """
        Get discriminator

        Returns:
            object: discriminator object
        """
        return self._discriminator

    @discriminator_net.setter
    def discriminator_net(self, net):
        self._discriminator = net

    def get_label(self, x, detach=False):  # pylint: disable=arguments-differ
        """
        Get data sample labels, i.e. true or fake.

        Args:
            x (Union(numpy.ndarray, torch.Tensor)): Discriminator input, i.e. data sample.
            detach (bool): if None detach from torch tensor variable (optional)

        Returns:
            torch.Tensor: Discriminator output, i.e. data label
        """
        # pylint: disable=not-callable, no-member
        import torch

        if isinstance(x, torch.Tensor):
            pass
        else:
            from torch.autograd.variable import Variable

            x = torch.tensor(x, dtype=torch.float32)
            x = Variable(x)

        if detach:
            return self._discriminator.forward(x).detach().numpy()
        else:
            return self._discriminator.forward(x)

    def loss(self, x, y, weights=None):
        """
        Loss function

        Args:
            x (torch.Tensor): Discriminator output.
            y (torch.Tensor): Label of the data point
            weights (torch.Tensor): Data weights.

        Returns:
            torch.Tensor: Loss w.r.t to the generated data points.
        """
        from torch import nn

        if weights is not None:
            loss_funct = nn.BCELoss(weight=weights, reduction="sum")
        else:
            loss_funct = nn.BCELoss()

        return loss_funct(x, y)

    def gradient_penalty(self, x, lambda_=5.0, k=0.01, c=1.0):
        """
        Compute gradient penalty for discriminator optimization

        Args:
            x (numpy.ndarray): Generated data sample.
            lambda_ (float): Gradient penalty coefficient 1.
            k (float): Gradient penalty coefficient 2.
            c (float): Gradient penalty coefficient 3.

        Returns:
            torch.Tensor: Gradient penalty.
        """
        # pylint: disable=not-callable, no-member
        import torch

        if isinstance(x, torch.Tensor):
            pass
        else:
            from torch.autograd.variable import Variable

            x = torch.tensor(x, dtype=torch.float32)
            x = Variable(x)
        # pylint: disable=no-member
        delta_ = torch.rand(x.size()) * c
        z = torch.autograd.Variable(x + delta_, requires_grad=True)
        o_l = self.get_label(z)
        # pylint: disable=no-member
        d_g = torch.autograd.grad(o_l, z, grad_outputs=torch.ones(o_l.size()), create_graph=True)[
            0
        ].view(z.size(0), -1)

        return lambda_ * ((d_g.norm(p=2, dim=1) - k) ** 2).mean()

    def train(
        self,
        data: Iterable,
        weights: Iterable,
        penalty: bool = False,
        quantum_instance: Optional[QuantumInstance] = None,
        shots: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform one training step w.r.t to the discriminator's parameters

        Args:
            data: Data batch.
            weights: Data sample weights.
            penalty: Indicate whether or not penalty function
               is applied to the loss function. Ignored if no penalty function defined.
            quantum_instance (QuantumInstance): used to run Quantum network.
               Ignored for a classical network.
            shots: Number of shots for hardware or qasm execution.
                Ignored for classical network

        Returns:
            dict: with discriminator loss and updated parameters.data, weights, penalty=True,
              quantum_instance=None, shots=None) -> Dict[str, Any]:
        """
        # pylint: disable=E1101
        # pylint: disable=E1102
        import torch
        from torch.autograd.variable import Variable

        # Reset gradients
        self._optimizer.zero_grad()
        real_batch = cast(Sequence, data)[0]
        real_prob = cast(Sequence, weights)[0]
        generated_batch = cast(Sequence, data)[1]
        generated_prob = cast(Sequence, weights)[1]

        real_batch = np.reshape(real_batch, (len(real_batch), self._n_features))
        real_batch = torch.tensor(real_batch, dtype=torch.float32)
        real_batch = Variable(real_batch)
        real_prob = np.reshape(real_prob, (len(real_prob), 1))
        real_prob = torch.tensor(real_prob, dtype=torch.float32)

        # Train on Real Data
        prediction_real = self.get_label(real_batch)

        # Calculate error and back propagate
        error_real = self.loss(prediction_real, torch.ones(len(prediction_real), 1), real_prob)
        error_real.backward()

        # Train on Generated Data
        generated_batch = np.reshape(generated_batch, (len(generated_batch), self._n_features))
        generated_prob = np.reshape(generated_prob, (len(generated_prob), 1))
        generated_prob = torch.tensor(generated_prob, dtype=torch.float32)
        prediction_fake = self.get_label(generated_batch)

        # Calculate error and back propagate
        error_fake = self.loss(
            prediction_fake, torch.zeros(len(prediction_fake), 1), generated_prob
        )
        error_fake.backward()

        if penalty:
            self.gradient_penalty(real_batch).backward()
        # pylint: enable=E1101
        # pylint: enable=E1102
        # Update weights with gradients
        self._optimizer.step()

        # Return error and predictions for real and fake inputs
        loss_ret = 0.5 * (error_real + error_fake)
        self._ret["loss"] = loss_ret.detach().numpy()
        params = []

        for param in self._discriminator.parameters():
            params.append(param.data.detach().numpy())
        self._ret["params"] = params

        return self._ret
