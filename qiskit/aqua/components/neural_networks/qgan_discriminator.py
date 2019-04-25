# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np
import os
import importlib

import logging
logger = logging.getLogger(__name__)

from qiskit.aqua import AquaError
from qiskit.aqua import Pluggable
from qiskit.aqua.components.neural_networks import NeuralNetwork

try:
    import torch
    from torch import nn, optim
    from torch.autograd.variable import Variable
    torch_loaded = True

except ImportError:
    logger.info('Pytorch is not installed. For installation instructions see https://pytorch.org/get-started/locally/')
    torch_loaded = False
    # raise Exception('Please install PyTorch')


class DiscriminatorNet(torch.nn.Module):
    """
    Discriminator
    """

    def __init__(self, n_features=1):
        """
        Initialize the discriminator network.
        Args:
            n_features: int, Dimension of input data samples.
        """

        super(DiscriminatorNet, self).__init__()
        self.n_features = n_features
        n_out = 1

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

    def forward(self, x):
        """

        Args:
            x: torch.Tensor, Discriminator input, i.e. data sample.

        Returns: torch.Tensor, Discriminator output, i.e. data label.

        """
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.out(x)

        return x


class Discriminator(NeuralNetwork):

    def __init__(self, n_features=1, discriminator_net=None, optimizer=None):
        """
        Initialize the discriminator.
        Args:
            n_features: int, Dimension of input data vector.
            discriminator_net: torch.nn.Module or None, Discriminator network.
            optimizer: torch.optim.Optimizer or None, Optimizer initialized w.r.t discriminator network parameters.
        """

        if not torch_loaded:
            raise AquaError('Pytorch is not installed. For installation instructions see '
                            'https://pytorch.org/get-started/locally/')

        self.n_features = n_features
        if isinstance(optimizer, optim.Optimizer):
            self._optimizer = optimizer
            if isinstance(discriminator_net, torch.nn.Module):
                self.discriminator = discriminator_net
            else:
                self.discriminator = DiscriminatorNet(self.n_features)
                self._optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-5, amsgrad=True)
        else:
            if isinstance(discriminator_net, torch.nn.Module):
                self.discriminator = discriminator_net
            else:
                self.discriminator = DiscriminatorNet(self.n_features)
            self._optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-5, amsgrad=True)

        self._ret = {}

    @classmethod
    def get_section_key_name(cls):
        return Pluggable.SECTION_KEY_NEURAL_NETWORK

    @staticmethod
    def check_pluggable_valid():
        err_msg = 'Pytorch is not installed. For installation instructions see https://pytorch.org/get-started/locally/'
        try:
            spec = importlib.util.find_spec('torch.Tensor')
            if spec is not None:
                spec = importlib.util.find_spec('torch.nn')
                if spec is not None:
                    return
        except Exception as e:
            logger.debug('{} {}'.format(err_msg, str(e)))
            raise AquaError(err_msg) from e

        raise AquaError(err_msg)

    def set_seed(self, seed):
        """
        Set seed.
        Args:
            seed: int, seed

        Returns:

        """
        torch.manual_seed(seed)
        return

    def save_model(self, snapshot_dir):
        """
        Save discriminator model
        Args:
            snapshot_dir: str, directory path for saving the model

        Returns:

        """
        torch.save(self.discriminator, os.path.join(snapshot_dir, 'discriminator.pt'))
        return

    def get_output(self, x):
        """
        Get data sample labels, i.e. true or fake.
        Args:
            x: numpy array or torch.Tensor, Discriminator input, i.e. data sample.

        Returns:torch.Tensor, Discriminator output, i.e. data label

        """

        if isinstance(x, torch.Tensor):
            pass
        else:
            x = torch.tensor(x, dtype=torch.float32)
            x = Variable(x)

        return self.discriminator.forward(x)

    def loss(self, x, y, weights=None):
        """
        Loss function
        Args:
            x: torch.Tensor, Discriminator output.
            y: torch.Tensor, Label of the data point
            weights: torch.Tensor, Data weights.

        Returns:torch.Tensor, Loss w.r.t to the generated data points.

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
            x: numpy array, Generated data sample.
            lambda_: float, Gradient penalty coefficient 1.
            k: float, Gradient penalty coefficient 2.
            c: float, Gradient penalty coefficient 3.

        Returns: torch.Tensor, Gradient penalty.

        """

        if isinstance(x, torch.Tensor):
            pass
        else:
            x = torch.tensor(x, dtype=torch.float32)
            x = Variable(x)
        delta_ = torch.rand(x.size()) * c
        z = Variable(x+delta_, requires_grad = True)
        o = self.get_output(z)
        d = torch.autograd.grad(o, z, grad_outputs=torch.ones(o.size()), create_graph=True)[0].view(z.size(0), -1)

        return lambda_ * ((d.norm(p=2,dim=1) - k)**2).mean()

    def train(self, real_batch, generated_batch, generated_prob, penalty=False):
        """
        Perform one training step w.r.t to the discriminator's parameters
        Args:
            real_batch: torch.Tensor, Training data batch.
            generated_batch: numpy array, Generated data batch.
            generated_prob: numpy array, Weights of the generated data samples, i.e. measurement frequency for
        qasm/hardware backends resp. measurement probability for statevector backend.
            penalty: Boolean, Indicate whether or not penalty function is applied to the loss function.

        Returns: dict, with Discriminator loss (torch.Tensor) and updated parameters (array).

        """

        # Reset gradients
        self._optimizer.zero_grad()

        real_batch = torch.tensor(real_batch, dtype=torch.float32)
        real_batch = Variable(real_batch)

        # Train on Real Data
        prediction_real = self.get_output(real_batch)

        # Calculate error and backpropagate
        error_real = self.loss(prediction_real, torch.ones(len(prediction_real), 1))
        error_real.backward()

        # Train on Generated Data
        generated_batch = np.reshape(generated_batch,(len(generated_batch), self.n_features))
        generated_prob = np.reshape(generated_prob, (len(generated_prob), 1))
        generated_prob = torch.tensor(generated_prob, dtype=torch.float32)
        prediction_fake = self.get_output(generated_batch)

        # Calculate error and backpropagate
        error_fake = self.loss(prediction_fake, torch.zeros(len(prediction_fake),1), generated_prob)
        error_fake.backward()

        if penalty:
            self.gradient_penalty(real_batch).backward()

        # Update weights with gradients
        self._optimizer.step()

        # Return error and predictions for real and fake inputs
        self._ret['loss'] = 0.5*(error_real + error_fake)
        params = []
        for param in self.discriminator.parameters():
            params.append(param.data.detach().numpy())
        self._ret['params'] = params

        return self._ret
