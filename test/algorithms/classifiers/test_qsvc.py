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

""" Test QSVC """

import unittest

from test import QiskitMachineLearningTestCase

import numpy as np

from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC, QuantumKernelTrainer
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.exceptions import QiskitMachineLearningError


def generate_tunable_feature_map():
    """
    Create a 2 qubit circuit consisting of 2 free parameters and 2 data bound parameters.
    """
    data_block = ZZFeatureMap(2)
    tunable_block = ZZFeatureMap(2)
    free_parameters = tunable_block.parameters

    for i in range(len(free_parameters)):
        free_parameters[i]._name = f"θ[{i}]"

    feature_map = data_block.compose(tunable_block).compose(data_block)

    return feature_map, free_parameters


def generate_data():
    X_train = np.array(
        [
            [0.9567612067697236, 0.29073480898094595],
            [0.5346295164276453, 0.28314590337282675],
            [0.4505164537165349, 0.1348844084345856],
            [0.717095964284348, 0.6926539694817647],
            [0.6899904242073004, 0.11894254875732169],
            [0.9331929543471993, 0.1710356466941273],
            [0.9465065903873696, 0.08866785584134762],
            [0.020821616675430255, 0.795816864607097],
            [0.11396873060321688, 0.597495651509774],
            [0.43382947964712637, 0.015927416849493836],
            [0.9442809322211615, 0.9895496473858371],
            [0.04664013387835264, 0.2098641515156191],
            [0.8170482966873857, 0.12136924758892997],
            [0.8674239716401573, 0.9695260771109943],
            [0.7819845649388245, 0.36615922941017176],
            [0.2616107759284595, 0.7224424998541275],
        ]
    )
    X_test = np.array(
        [
            [0.9370643995995782, 0.6308179906118104],
            [0.5942319745353428, 0.6530442138654083],
            [0.699064828319063, 0.5601537947326779],
            [0.0705901741870586, 0.2036948590213834],
        ]
    )

    y_train = np.array(
        [
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
        ]
    )
    y_test = np.array([1.0, 1.0, -1.0, -1.0])

    return X_train, y_train, X_test, y_test


class TestQSVC(QiskitMachineLearningTestCase):
    """Test QSVC Algorithm"""

    def setUp(self):
        super().setUp()

        algorithm_globals.random_seed = 10598

        self.statevector_simulator = QuantumInstance(
            BasicAer.get_backend("statevector_simulator"),
            shots=1,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        self.feature_map = ZZFeatureMap(feature_dimension=2, reps=2)

        self.sample_train = np.asarray(
            [
                [3.07876080, 1.75929189],
                [6.03185789, 5.27787566],
                [6.22035345, 2.70176968],
                [0.18849556, 2.82743339],
            ]
        )
        self.label_train = np.asarray([0, 0, 1, 1])

        self.sample_test = np.asarray([[2.199114860, 5.15221195], [0.50265482, 0.06283185]])
        self.label_test = np.asarray([0, 1])

    def test_qsvc(self):
        """Test QSVC"""
        qkernel = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        qsvc = QSVC(quantum_kernel=qkernel)
        qsvc.fit(self.sample_train, self.label_train)
        score = qsvc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 0.5)

    def test_empty_kernel(self):
        """Test QSVC with empty QuantumKernel"""
        qkernel = QuantumKernel()
        qsvc = QSVC(quantum_kernel=qkernel)

        with self.assertRaises(QiskitMachineLearningError):
            _ = qsvc.fit(self.sample_train, self.label_train)

    def test_change_kernel(self):
        """Test QSVC with QuantumKernel later"""
        qkernel = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        qsvc = QSVC()
        qsvc.quantum_kernel = qkernel
        qsvc.fit(self.sample_train, self.label_train)
        score = qsvc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 0.5)

    def test_qsvc_parameters(self):
        """Test QSVC with extra constructor parameters"""
        qkernel = QuantumKernel(
            feature_map=self.feature_map, quantum_instance=self.statevector_simulator
        )

        qsvc = QSVC(quantum_kernel=qkernel, tol=1e-4, C=0.5)
        qsvc.fit(self.sample_train, self.label_train)
        score = qsvc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 0.5)

    def test_unbound_free_params(self):
        """Test QSVC with extra constructor parameters"""
        fm, fp = generate_tunable_feature_map()
        X_train, y_train, X_test, y_test = generate_data()
        qkernel = QuantumKernel(
            feature_map=fm,
            free_parameters=fp,
            quantum_instance=BasicAer.get_backend("qasm_simulator"),
        )

        qkt = QuantumKernelTrainer(qkernel)

        qsvc = QSVC(quantum_kernel=qkt)
        qsvc.fit(X_train, y_train)
        score = qsvc.score(X_test, y_test)

        self.assertEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
