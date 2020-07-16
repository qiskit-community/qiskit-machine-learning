# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Generative Adversarial Network."""

from typing import Optional, Union, List, Dict, Any
import csv
import os
import logging

import numpy as np
from scipy.stats import entropy

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance, AquaError, aqua_globals
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.components.neural_networks.discriminative_network import DiscriminativeNetwork
from qiskit.aqua.components.neural_networks.generative_network import GenerativeNetwork
from qiskit.aqua.components.neural_networks.quantum_generator import QuantumGenerator
from qiskit.aqua.components.neural_networks.numpy_discriminator import NumPyDiscriminator
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.components.uncertainty_models import UnivariateVariationalDistribution
from qiskit.aqua.components.uncertainty_models import MultivariateVariationalDistribution
from qiskit.aqua.utils.dataset_helper import discretize_and_truncate
from qiskit.aqua.utils.validation import validate_min

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class QGAN(QuantumAlgorithm):
    """The Quantum Generative Adversarial Network algorithm.

    The qGAN [1] is a hybrid quantum-classical algorithm used for generative modeling tasks.

    This adaptive algorithm uses the interplay of a generative
    :class:`~qiskit.aqua.components.neural_networks.GenerativeNetwork` and a
    discriminative :class:`~qiskit.aqua.components.neural_networks.DiscriminativeNetwork`
    network to learn the probability distribution underlying given training data.

    These networks are trained in alternating optimization steps, where the discriminator tries to
    differentiate between training data samples and data samples from the generator and the
    generator aims at generating samples which the discriminator classifies as training data
    samples. Eventually, the quantum generator learns the training data's underlying probability
    distribution. The trained quantum generator loads a quantum state which is a model of the
    target distribution.

    **References:**

    [1] Zoufal et al.,
        `Quantum Generative Adversarial Networks for learning and loading random distributions
        <https://www.nature.com/articles/s41534-019-0223-2>`_
    """

    def __init__(self, data: np.ndarray, bounds: Optional[np.ndarray] = None,
                 num_qubits: Optional[np.ndarray] = None, batch_size: int = 500,
                 num_epochs: int = 3000, seed: int = 7,
                 discriminator: Optional[DiscriminativeNetwork] = None,
                 generator: Optional[GenerativeNetwork] = None,
                 tol_rel_ent: Optional[float] = None, snapshot_dir: Optional[str] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None) -> None:
        """

        Args:
            data: Training data of dimension k
            bounds: k min/max data values [[min_0,max_0],...,[min_k-1,max_k-1]]
                if univariate data: [min_0,max_0]
            num_qubits: k numbers of qubits to determine representation resolution,
                i.e. n qubits enable the representation of 2**n values
                [num_qubits_0,..., num_qubits_k-1]
            batch_size: Batch size, has a min. value of 1.
            num_epochs: Number of training epochs
            seed: Random number seed
            discriminator: Discriminates between real and fake data samples
            generator: Generates 'fake' data samples
            tol_rel_ent: Set tolerance level for relative entropy.
                If the training achieves relative entropy equal or lower than tolerance it finishes.
            snapshot_dir: Directory in to which to store cvs file with parameters,
                if None (default) then no cvs file is created.
            quantum_instance: Quantum Instance or Backend
        Raises:
            AquaError: invalid input
        """
        validate_min('batch_size', batch_size, 1)
        super().__init__(quantum_instance)
        if data is None:
            raise AquaError('Training data not given.')
        self._data = np.array(data)
        if bounds is None:
            bounds_min = np.percentile(self._data, 5, axis=0)
            bounds_max = np.percentile(self._data, 95, axis=0)
            bounds = []
            for i, _ in enumerate(bounds_min):
                bounds.append([bounds_min[i], bounds_max[i]])
        if np.ndim(data) > 1:
            if len(bounds) != (len(num_qubits) or len(data[0])):
                raise AquaError('Dimensions of the data, the length of the data bounds '
                                'and the numbers of qubits per '
                                'dimension are incompatible.')
        else:
            if (np.ndim(bounds) or len(num_qubits)) != 1:
                raise AquaError('Dimensions of the data, the length of the data bounds '
                                'and the numbers of qubits per '
                                'dimension are incompatible.')
        self._bounds = np.array(bounds)
        self._num_qubits = num_qubits
        # pylint: disable=unsubscriptable-object
        if np.ndim(data) > 1:
            if self._num_qubits is None:
                self._num_qubits = np.ones[len(data[0])] * 3
        else:
            if self._num_qubits is None:
                self._num_qubits = np.array([3])
        self._data, self._data_grid, self._grid_elements, self._prob_data = \
            discretize_and_truncate(self._data, self._bounds, self._num_qubits,
                                    return_data_grid_elements=True,
                                    return_prob=True, prob_non_zero=True)
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._snapshot_dir = snapshot_dir
        self._g_loss = []  # type: List[float]
        self._d_loss = []  # type: List[float]
        self._rel_entr = []  # type: List[float]
        self._tol_rel_ent = tol_rel_ent

        self._random_seed = seed

        if generator is None:
            self.set_generator()
        else:
            self._generator = generator
        if discriminator is None:
            self.set_discriminator()
        else:
            self._discriminator = discriminator

        self.seed = self._random_seed

        self._ret = {}  # type: Dict[str, Any]

    @property
    def seed(self):
        """ Returns random seed """
        return self._random_seed

    @seed.setter
    def seed(self, s):
        """
        Sets the random seed for QGAN and updates the aqua_globals seed
        at the same time

        Args:
            s (int): random seed
        """
        self._random_seed = s
        aqua_globals.random_seed = self._random_seed
        self._discriminator.set_seed(self._random_seed)

    @property
    def tol_rel_ent(self):
        """ Returns tolerance for relative entropy """
        return self._tol_rel_ent

    @tol_rel_ent.setter
    def tol_rel_ent(self, t):
        """
        Set tolerance for relative entropy

        Args:
            t (float): or None, Set tolerance level for relative entropy.
                If the training achieves relative
                entropy equal or lower than tolerance it finishes.
        """
        self._tol_rel_ent = t

    @property
    def generator(self):
        """ Returns generator """
        return self._generator

    # pylint: disable=unused-argument
    def set_generator(self, generator_circuit: Optional[Union[QuantumCircuit,
                                                              UnivariateVariationalDistribution,
                                                              MultivariateVariationalDistribution]
                                                        ] = None,
                      generator_init_params: Optional[np.ndarray] = None,
                      generator_optimizer: Optional[Optimizer] = None):
        """Initialize generator.

        Args:
            generator_circuit: parameterized quantum circuit which sets
                the structure of the quantum generator
            generator_init_params: initial parameters for the generator circuit
            generator_optimizer: optimizer to be used for the training of the generator
        """
        self._generator = QuantumGenerator(self._bounds, self._num_qubits,
                                           generator_circuit, generator_init_params,
                                           self._snapshot_dir)

    @property
    def discriminator(self):
        """ Returns discriminator """
        return self._discriminator

    def set_discriminator(self, discriminator=None):
        """
        Initialize discriminator.

        Args:
            discriminator (Discriminator): discriminator
        """

        if discriminator is None:
            self._discriminator = NumPyDiscriminator(len(self._num_qubits))
        else:
            self._discriminator = discriminator
        self._discriminator.set_seed(self._random_seed)

    @property
    def g_loss(self) -> List[float]:
        """ Returns generator loss """
        return self._g_loss

    @property
    def d_loss(self) -> List[float]:
        """ Returns discriminator loss """
        return self._d_loss

    @property
    def rel_entr(self) -> List[float]:
        """ Returns relative entropy between target and trained distribution """
        return self._rel_entr

    def get_rel_entr(self) -> float:
        """ Get relative entropy between target and trained distribution """
        samples_gen, prob_gen = self._generator.get_output(self._quantum_instance)
        temp = np.zeros(len(self._grid_elements))
        for j, sample in enumerate(samples_gen):
            for i, element in enumerate(self._grid_elements):
                if sample == element:
                    temp[i] += prob_gen[j]
        prob_gen = temp
        prob_gen = [1e-8 if x == 0 else x for x in prob_gen]
        rel_entr = entropy(prob_gen, self._prob_data)
        return rel_entr

    def _store_params(self, e, d_loss, g_loss, rel_entr):
        with open(os.path.join(self._snapshot_dir, 'output.csv'), mode='a') as csv_file:
            fieldnames = ['epoch', 'loss_discriminator',
                          'loss_generator', 'params_generator', 'rel_entropy']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'epoch': e, 'loss_discriminator': np.average(d_loss),
                             'loss_generator': np.average(g_loss), 'params_generator':
                                 self._generator.generator_circuit.params, 'rel_entropy': rel_entr})
        self._discriminator.save_model(self._snapshot_dir)  # Store discriminator model

    def train(self):
        """
        Train the qGAN

        Raises:
            AquaError: Batch size bigger than the number of items in the truncated data set
        """
        if self._snapshot_dir is not None:
            with open(os.path.join(self._snapshot_dir, 'output.csv'), mode='w') as csv_file:
                fieldnames = ['epoch', 'loss_discriminator', 'loss_generator', 'params_generator',
                              'rel_entropy']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

        if len(self._data) < self._batch_size:
            raise AquaError(
                'The batch size needs to be less than the '
                'truncated data size of {}'.format(len(self._data)))

        for e in range(self._num_epochs):
            aqua_globals.random.shuffle(self._data)
            index = 0
            while (index + self._batch_size) <= len(self._data):
                real_batch = self._data[index: index + self._batch_size]
                index += self._batch_size
                generated_batch, generated_prob = self._generator.get_output(self._quantum_instance,
                                                                             shots=self._batch_size)

                # 1. Train Discriminator
                ret_d = self._discriminator.train([real_batch, generated_batch],
                                                  [np.ones(len(real_batch)) / len(real_batch),
                                                   generated_prob])
                d_loss_min = ret_d['loss']

                # 2. Train Generator
                self._generator.set_discriminator(self._discriminator)
                ret_g = self._generator.train(self._quantum_instance, shots=self._batch_size)
                g_loss_min = ret_g['loss']

            self._d_loss.append(np.around(float(d_loss_min), 4))
            self._g_loss.append(np.around(g_loss_min, 4))

            rel_entr = self.get_rel_entr()
            self._rel_entr.append(np.around(rel_entr, 4))
            self._ret['params_d'] = ret_d['params']
            self._ret['params_g'] = ret_g['params']
            self._ret['loss_d'] = np.around(float(d_loss_min), 4)
            self._ret['loss_g'] = np.around(g_loss_min, 4)
            self._ret['rel_entr'] = np.around(rel_entr, 4)

            if self._snapshot_dir is not None:
                self._store_params(e, np.around(d_loss_min, 4),
                                   np.around(g_loss_min, 4), np.around(rel_entr, 4))
            logger.debug('Epoch %s/%s...', e + 1, self._num_epochs)
            logger.debug('Loss Discriminator: %s', np.around(float(d_loss_min), 4))
            logger.debug('Loss Generator: %s', np.around(g_loss_min, 4))
            logger.debug('Relative Entropy: %s', np.around(rel_entr, 4))

            if self._tol_rel_ent is not None:
                if rel_entr <= self._tol_rel_ent:
                    break

    def _run(self):
        """
        Run qGAN training

        Returns:
            dict: with generator(discriminator) parameters & loss, relative entropy
        Raises:
            AquaError: invalid backend
        """
        if self._quantum_instance.backend_name == ('unitary_simulator' or 'clifford_simulator'):
            raise AquaError(
                'Chosen backend not supported - '
                'Set backend either to statevector_simulator, qasm_simulator'
                ' or actual quantum hardware')
        self.train()

        return self._ret
