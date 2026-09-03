# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2026.
# (C) Copyright UKRI-STFC (Hartree Centre) 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Pegasos QKA"""

import unittest
from unittest.mock import patch

from test import QiskitMachineLearningTestCase

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import z_feature_map
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.circuit import ParameterVector

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms import PegasosQKA
from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel
from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.state_fidelities import ComputeUncompute


class TestPegasosQKA(QiskitMachineLearningTestCase):
    """Test Pegasos QKA Algorithm"""

    def setUp(self):
        super().setUp()

        algorithm_globals.random_seed = 10598

        # number of qubits is equal to the number of features
        self.q = 2
        # number of steps performed during the training procedure
        self.tau = 100

        z_map = z_feature_map(feature_dimension=self.q, reps=1)

        # Create a rotational layer to train.
        self.training_params = ParameterVector("θ", 1)
        fm = QuantumCircuit(2)
        fm.ry(self.training_params[0], 0)
        fm.ry(self.training_params[0], 1)

        # Create the feature map, composed of our two circuits
        self.feature_map = fm.compose(z_map)

        self.sampler = Sampler()
        self.fidelity = ComputeUncompute(sampler=self.sampler)
        self.qkernel = TrainableFidelityQuantumKernel(
            fidelity=self.fidelity,
            feature_map=self.feature_map,
            training_parameters=self.training_params,
        )

        sample, label = make_blobs(
            n_samples=20, n_features=2, centers=2, random_state=3, shuffle=True
        )
        sample = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(sample)

        # split into train and test set
        self.sample_train = sample[:15]
        self.label_train = label[:15]
        self.sample_test = sample[15:]
        self.label_test = label[15:]

    def test_qka(self):
        """Test PegasosQKA"""

        pegasos_qka = PegasosQKA(quantum_kernel=self.qkernel, C=1000, num_steps=self.tau)

        pegasos_qka.fit(self.sample_train, self.label_train)
        score = pegasos_qka.score(self.sample_test, self.label_test)

        self.assertEqual(score, 1.0)

    def test_constructor(self):
        """Tests properties of PegasosQKA"""
        with self.subTest("Default parameters"):
            with self.assertRaises(QiskitMachineLearningError):
                pegasos_qka = PegasosQKA()

        with self.subTest("Default initial thetas"):
            pegasos_qka = PegasosQKA(
                quantum_kernel=self.qkernel,
                C=1000,
                num_steps=self.tau,
            )

            np.testing.assert_array_equal(
                pegasos_qka._theta,
                np.zeros(len(self.training_params)),
            )

        with self.subTest("Specified initial thetas"):
            initial_thetas = np.full(len(self.training_params), 0.5)

            pegasos_qka = PegasosQKA(
                quantum_kernel=self.qkernel,
                C=1000,
                num_steps=self.tau,
                initial_thetas=initial_thetas,
            )

            np.testing.assert_array_equal(
                pegasos_qka._theta,
                initial_thetas,
            )

        with self.subTest("Incorrect initial theta"):
            with self.assertRaises(ValueError):
                PegasosQKA(
                    quantum_kernel=self.qkernel,
                    initial_thetas=np.zeros(len(self.training_params) + 1),
                )

        with self.subTest("PegasosQKA with TrainableFidelityQuantumKernel"):

            pegasos_qka = PegasosQKA(quantum_kernel=self.qkernel, C=1000, num_steps=self.tau)
            self.assertIsInstance(pegasos_qka.quantum_kernel, TrainableFidelityQuantumKernel)
            self.assertEqual(
                pegasos_qka.quantum_kernel.num_training_parameters,
                len(self.training_params),
            )

    def test_theta_update(self):
        """Test theta update."""

        pegasos_qka = PegasosQKA(
            quantum_kernel=self.qkernel,
            C=1,
            num_steps=self.tau,
            learning_rate=0.1,
            perturbations=0.1,
        )  # theta -> all zeros

        pegasos_qka._label_map = {0: -1, 1: 1}
        pegasos_qka._alphas = {}

        # set bernoulli_perturbation
        perturbation = np.ones(len(self.training_params))

        with patch(
            "qiskit_machine_learning.algorithms.classifiers.pegasos_qka.bernoulli_perturbation",
            return_value=perturbation,
        ):
            # value = self._compute_weighted_kernel_sum(index, X, training=True)
            # obj = -factor * self._compute_weighted_kernel_sum(index, X, training=True)
            pegasos_qka._compute_weighted_kernel_sum = unittest.mock.Mock(
                side_effect=[0.0, 0.2, 0.0]  # value, obj_plus, obj_minus
            )

            pegasos_qka._update_step(
                0, self.sample_train, np.ones(len(self.sample_train), dtype=int), 1
            )
            # check (y_step * self.C / step) * value < 1 --> True
            # factor = y_step * self.C / step
            # gradient = (obj_plus - obj_minus) / (2 * learning_rate)

        # new_theta = new_theta - learning_rate * gradient * perturbation
        expected_theta = np.full(len(self.training_params), 0.1)
        np.testing.assert_array_equal(pegasos_qka._theta, expected_theta)

        # self._alphas[index] = self._alphas.get(index, 0) + 1
        self.assertEqual(pegasos_qka._alphas[0], 1)

        # self._support_thetas[index].append(support_theta)
        np.testing.assert_array_equal(
            pegasos_qka._support_thetas[0][0],
            np.zeros(len(self.training_params)),
        )

    def test_no_theta_update(self):
        """Test theta is not updated when margin condition is satisfied."""

        pegasos_qka = PegasosQKA(
            quantum_kernel=self.qkernel,
            C=1,
            num_steps=self.tau,
            learning_rate=0.1,
        )

        pegasos_qka._label_map = {0: -1, 1: 1}
        pegasos_qka._alphas = {}

        initial_theta = pegasos_qka._theta.copy()

        pegasos_qka._compute_weighted_kernel_sum = unittest.mock.Mock(return_value=2.0)

        pegasos_qka._update_step(
            0,
            self.sample_train,
            np.ones(len(self.sample_train), dtype=int),
            1,
        )
        # check (y_step * self.C / step) * value < 1 --> False
        # = no update
        np.testing.assert_array_equal(pegasos_qka._theta, initial_theta)
        self.assertEqual(pegasos_qka._alphas, {})
        self.assertEqual(pegasos_qka._support_thetas, {})

    def test_support_thetas(self):
        """Test support theta history."""

        pegasos_qka = PegasosQKA(
            quantum_kernel=self.qkernel,
            C=1,
            num_steps=self.tau,
            learning_rate=0.1,
            perturbations=0.1,
        )

        pegasos_qka._label_map = {0: -1, 1: 1}
        pegasos_qka._alphas = {}

        # set bernoulli_perturbation
        perturbation = np.ones(len(self.training_params))

        with patch(
            "qiskit_machine_learning.algorithms.classifiers.pegasos_qka.bernoulli_perturbation",
            return_value=perturbation,
        ):
            pegasos_qka._compute_weighted_kernel_sum = unittest.mock.Mock(
                side_effect=[
                    0.0,
                    0.2,
                    0.0,
                    0.0,
                    0.2,
                    0.0,
                ]
            )

            y = np.ones(len(self.sample_train), dtype=int)

            pegasos_qka._update_step(0, self.sample_train, y, 1)
            pegasos_qka._update_step(0, self.sample_train, y, 2)

        self.assertEqual(len(pegasos_qka._support_thetas[0]), 2)

        np.testing.assert_array_equal(
            pegasos_qka._support_thetas[0][0],
            np.zeros(len(self.training_params)),
        )

        np.testing.assert_array_equal(
            pegasos_qka._support_thetas[0][1],
            np.full(len(self.training_params), 0.1),
        )

        self.assertEqual(pegasos_qka._alphas[0], 2)

    def test_evaluate_kernel(self):
        """Test kernel evaluation over support theta history."""

        pegasos_qka = PegasosQKA(
            quantum_kernel=self.qkernel,
            C=1,
            num_steps=self.tau,
        )

        pegasos_qka._left_theta = np.zeros(len(self.training_params))
        pegasos_qka._kernel_offset = 1
        support_indices = [0, 1]

        pegasos_qka._support_thetas = {
            0: [
                np.zeros(len(self.training_params)),
                np.full(len(self.training_params), 0.1),
            ],
            1: [
                np.full(len(self.training_params), 0.2),
            ],
        }

        pegasos_qka._quantum_kernel.evaluate = unittest.mock.Mock(side_effect=[0.2, 0.4, 0.8])

        values = pegasos_qka._evaluate_kernel(
            self.sample_test[0],
            self.sample_train[:2],
            support_indices,
        )

        expected = np.array([1.3, 1.8])

        np.testing.assert_allclose(values, expected)
        self.assertEqual(pegasos_qka._quantum_kernel.evaluate.call_count, 3)
