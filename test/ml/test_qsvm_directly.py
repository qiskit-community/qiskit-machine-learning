# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test QSVM Directly """

from test.ml.common import QiskitMLTestCase

from qiskit.ml.datasets import ad_hoc_data
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.utils import split_dataset_to_data_and_labels


class TestQSVMDirectly(QiskitMLTestCase):
    """QSVM Directly tests."""

    def test_qsvm_directly(self):
        """QSVM Directly test."""

        feature_dim = 2  # dimension of each data point
        aqua_globals.random_seed = 10598

        _, training_input, test_input, _ = ad_hoc_data(training_size=20,
                                                       test_size=10,
                                                       n=feature_dim,
                                                       gap=0.3,
                                                       plot_data=False)
        _, class_to_label = split_dataset_to_data_and_labels(test_input)
        self.assertEqual(class_to_label, {'A': 0, 'B': 1})

        feature_map = SecondOrderExpansion(feature_dimension=feature_dim,
                                           depth=2,
                                           entangler_map=[[0, 1]])
        svm = QSVM(feature_map, training_input, test_input, None)
        svm.random_seed = aqua_globals.random_seed
        quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                           shots=1024,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = svm.run(quantum_instance)
        self.log.debug(result['testing_accuracy'])
        self.assertGreaterEqual(result['testing_accuracy'], 0.5)
