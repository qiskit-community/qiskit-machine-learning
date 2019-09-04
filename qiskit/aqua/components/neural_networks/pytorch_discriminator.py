# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Discriminator
"""

import os
import importlib
import logging

import numpy as np

from qiskit.aqua import AquaError, Pluggable
from .discriminative_network import DiscriminativeNetwork

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name

try:
    import torch
    from torch import nn, optim
    from torch.autograd.variable import Variable
    torch_loaded = True
except ImportError:
    logger.info('Pytorch is not installed. For installation instructions '
                'see https://pytorch.org/get-started/locally/')
    torch_loaded = False


class DiscriminatorNet(torch.nn.Module):
    """
    Discriminator
    """

    def __init__(self, n_features=1, n_out=1):
        """
        Initialize the discriminator network.
        Args:
            n_features (int): Dimension of input data samples.
            n_out (int): n out
        """

        super(DiscriminatorNet, self).__init__()
        self.n_features = n_features

        self.hidden0 = nn.Sequential(
            nn.Linear(self.n_features, 512),
            nn.LeakyReLU(0.2),
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
        )
        self.out = nn.Sequential(
            nn.Linear(256, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):  # pylint: disable=arguments-differ
        """

        Args:
            x (torch.Tensor): Discriminator input, i.e. data sample.

        Returns:
            torch.Tensor: Discriminator output, i.e. data label.

        """
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.out(x)

        return x


class ClassicalDiscriminator(DiscriminativeNetwork):
    """
        Discriminator
    """
    CONFIGURATION = {
        'name': 'PytorchDiscriminator',
        'description': 'qGAN Discriminator Network',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'discriminator_schema',
            'type': 'object',
            'properties': {
                'n_features': {
                    'type': 'integer',
                    'default': 1
                },
                'n_out': {
                    'type': 'integer',
                    'default': 1
                }

            },
            'additionalProperties': False
        }
    }

    def __init__(self, n_features=1, n_out=1):
        """
        Initialize the discriminator.
        Args:
            n_features (int): Dimension of input data vector.
            n_out (int):, Dimension of the discriminator's output vector.
        Raises:
            AquaError: Pytorch not installed

        """
        super().__init__()
        if not torch_loaded:
            raise AquaError('Pytorch is not installed. For installation instructions see '
                            'https://pytorch.org/get-started/locally/')

        self._n_features = n_features
        self._n_out = n_out
        # discriminator_net: torch.nn.Module or None, Discriminator network.
        self._discriminator = DiscriminatorNet(self._n_features, self._n_out)
        # optimizer: torch.optim.Optimizer or None, Optimizer initialized w.r.t
        # discriminator network parameters.
        self._optimizer = optim.Adam(self._discriminator.parameters(), lr=1e-5, amsgrad=True)

        self._ret = {}

    @classmethod
    def get_section_key_name(cls):
        return Pluggable.SECTION_KEY_DISCRIMINATIVE_NET

    @staticmethod
    def check_pluggable_valid():
        err_msg = \
            'Pytorch is not installed. For installation instructions ' \
            'see https://pytorch.org/get-started/locally/'
        try:
            spec = importlib.util.find_spec('torch.optim')
            if spec is not None:
                spec = importlib.util.find_spec('torch.nn')
                if spec is not None:
                    return
        except Exception as ex:  # pylint: disable=broad-except
            logger.debug('%s %s', err_msg, str(ex))
            raise AquaError(err_msg) from ex

        raise AquaError(err_msg)

    def set_seed(self, seed):
        """
        Set seed.
        Args:
            seed (int): seed
        """
        torch.manual_seed(seed)

    def save_model(self, snapshot_dir):
        """
        Save discriminator model
        Args:
            snapshot_dir (str):  directory path for saving the model
        """
        torch.save(self._discriminator, os.path.join(snapshot_dir, 'discriminator.pt'))

    def load_model(self, load_dir):
        """
        Save discriminator model
        Args:
            load_dir (str): file with stored pytorch discriminator model to be loaded
        """
        torch.load(load_dir)

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
        if isinstance(x, torch.Tensor):
            pass
        else:
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
        if weights is not None:
            loss_funct = nn.BCELoss(weight=weights, reduction='sum')
        else:
            loss_funct = nn.BCELoss()

        return loss_funct(x, y)

    def gradient_penalty(self, x, lambda_=5., k=0.01, c=1.):
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
        if isinstance(x, torch.Tensor):
            pass
        else:
            x = torch.tensor(x, dtype=torch.float32)
            x = Variable(x)
        # pylint: disable=no-member
        delta_ = torch.rand(x.size()) * c
        z = Variable(x+delta_, requires_grad=True)
        o = self.get_label(z)
        # pylint: disable=no-member
        d = torch.autograd.grad(o, z, grad_outputs=torch.ones(o.size()),
                                create_graph=True)[0].view(z.size(0), -1)

        return lambda_ * ((d.norm(p=2, dim=1) - k)**2).mean()

    def train(self, data, weights, penalty=True, quantum_instance=None, shots=None):
        """
        Perform one training step w.r.t to the discriminator's parameters
        Args:
            data (tuple):
                real_batch: torch.Tensor, Training data batch.
                generated_batch: numpy array, Generated data batch.
            weights (tuple): real problem, generated problem
            penalty (bool): Indicate whether or not penalty function is
                    applied to the loss function.
            quantum_instance (QuantumInstance): Quantum Instance (depreciated)
            shots (int): Number of shots for hardware or qasm execution.
                        Depreciated for classical network(depreciated)

        Returns:
            dict: with Discriminator loss (torch.Tensor) and updated parameters (array).

        """
        # pylint: disable=E1101
        # pylint: disable=E1102
        # Reset gradients
        self._optimizer.zero_grad()
        real_batch = data[0]
        real_prob = weights[0]
        generated_batch = data[1]
        generated_prob = weights[1]

        real_batch = np.reshape(real_batch, (len(real_batch), 1))
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
        error_fake = self.loss(prediction_fake, torch.zeros(len(prediction_fake), 1),
                               generated_prob)
        error_fake.backward()

        if penalty:
            self.gradient_penalty(real_batch).backward()
        # pylint: enable=E1101
        # pylint: enable=E1102
        # Update weights with gradients
        self._optimizer.step()

        # Return error and predictions for real and fake inputs
        loss_ret = 0.5 * (error_real + error_fake)
        self._ret['loss'] = loss_ret.detach().numpy()
        params = []

        for param in self._discriminator.parameters():
            params.append(param.data.detach().numpy())
        self._ret['params'] = params

        return self._ret
