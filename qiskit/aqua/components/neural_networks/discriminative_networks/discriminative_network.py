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


from abc import abstractmethod

from qiskit.aqua import Pluggable


class DiscriminativeNetwork(Pluggable):
    """Base class for discriminative Quantum or Classical Neural Networks.

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is
        available.

        Args:
            configuration (dict): configuration dictionary
    """
    @abstractmethod
    def __init__(self):
        super().__init__()
        self._num_parameters = 0
        self._num_qubits = 0
        self._bounds = list()
        pass

    @classmethod
    def init_params(cls, params):
        # We might to a differentiation between quantum and classical networks after all.
        # The QNN will take a variational form, (possibly) a quantum input state object and a qiskit optimizer
        # The CNN will take a torch.nn.Module, a torch.tensor and a torch.optimizer (if given as PyTorch object)

        return cls(**args)

    @classmethod
    @abstractmethod
    def get_section_key_name(cls):
        pass

    @abstractmethod
    def get_label(self, x):
        """ Apply quantum/classical neural network to the given input sample and compute the respective data label
        Args:
            x: Discriminator input, i.e. data sample.

        Returns: Computed data label

        """
        raise NotImplementedError()

    @abstractmethod
    def loss(self, x, y, weights=None):
        """Loss function used for optimization

        Args:
            x: Discriminator output.
            y: Label of the data point
            weights: Data weights.

        Returns: Loss w.r.t to the generated data points.

        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, real_batch, generated_batch, generated_prob, penalty=False, quantum_instance=None, shots = None):
        """
        Perform one training step w.r.t to the discriminator's parameters
        Args:
            real_batch: Training data batch.
            generated_batch: Generated data batch.
            generated_prob: Weights of the generated data samples, i.e. measurement frequency for
                            qasm/hardware backends resp. measurement probability for statevector backend.
            penalty: Boolean, Indicate whether or not penalty function is applied to the loss function.
                    If no penalty function defined - depreciate
                        quantum_instance: QuantumInstance, used to run the generator circuit.
                        Depreciated for classical network
            shots: int, Number of shots for hardware or qasm execution. Depreciated for classical network

        Returns: dict, with Discriminator loss and updated parameters.

        """
        raise NotImplementedError()
