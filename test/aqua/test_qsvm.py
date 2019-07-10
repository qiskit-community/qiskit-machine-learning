# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os

import numpy as np
from qiskit import BasicAer

from test.aqua.common import QiskitAquaTestCase
from qiskit.aqua import run_algorithm, QuantumInstance, aqua_globals
from qiskit.aqua.input import ClassificationInput
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.algorithms import QSVM


class TestQSVM(QiskitAquaTestCase):

    def setUp(self):
        super().setUp()
        self.random_seed = 10598
        self.shots = 12000
        np.random.seed(self.random_seed)
        self.training_data = {'A': np.asarray([[2.95309709, 2.51327412],
                                               [3.14159265, 4.08407045]]),
                              'B': np.asarray([[4.08407045, 2.26194671],
                                               [4.46106157, 2.38761042]])}
        self.testing_data = {'A': np.asarray([[3.83274304, 2.45044227]]),
                             'B': np.asarray([[3.89557489, 0.31415927]])}

        self.svm_input = ClassificationInput(self.training_data, self.testing_data)

    def test_qsvm_binary_via_run_algorithm(self):

        training_input = {'A': np.asarray([[0.6560706, 0.17605998], [0.14154948, 0.06201424],
                                           [0.80202323, 0.40582692], [0.46779595, 0.39946754],
                                           [0.57660199, 0.21821317]]),
                          'B': np.asarray([[0.38857596, -0.33775802], [0.49946978, -0.48727951],
                                           [-0.30119743, -0.11221681], [-0.16479252, -0.08640519],
                                           [0.49156185, -0.3660534]])}

        test_input = {'A': np.asarray([[0.57483139, 0.47120732], [0.48372348, 0.25438544],
                                       [0.08791134, 0.11515506], [0.45988094, 0.32854319],
                                       [0.53015085, 0.41539212]]),
                      'B': np.asarray([[-0.06048935, -0.48345293], [-0.01065613, -0.33910828],
                                       [-0.17323832, -0.49535592], [0.14043268, -0.87869109],
                                       [-0.15046837, -0.47340207]])}

        total_array = np.concatenate((test_input['A'], test_input['B']))

        params = {
            'problem': {'name': 'classification', 'random_seed': self.random_seed},
            'backend': {'shots': self.shots},
            'algorithm': {
                'name': 'QSVM'
            }
        }
        backend = BasicAer.get_backend('qasm_simulator')
        algo_input = ClassificationInput(training_input, test_input, total_array)
        result = run_algorithm(params, algo_input, backend=backend)
        self.assertEqual(result['testing_accuracy'], 0.6)
        self.assertEqual(result['predicted_classes'], ['A', 'A', 'A', 'A', 'A',
                                                       'A', 'B', 'A', 'A', 'A'])

    def test_qsvm_binary_directly(self):

        ref_kernel_training = np.array([[1., 0.85366667, 0.12341667, 0.36408333],
                                        [0.85366667, 1., 0.11141667, 0.45491667],
                                        [0.12341667, 0.11141667, 1., 0.667],
                                        [0.36408333, 0.45491667, 0.667, 1.]])

        ref_kernel_testing = np.array([[0.14316667, 0.18208333, 0.4785, 0.14441667],
                                       [0.33608333, 0.3765, 0.02316667, 0.15858333]])

        # ref_alpha = np.array([0.36064489, 1.49204209, 0.0264953, 1.82619169])
        ref_alpha = np.array([0.34903335, 1.48325498, 0.03074852, 1.80153981])
        # ref_bias = np.array([-0.03380763])
        ref_bias = np.array([-0.03059226])

        ref_support_vectors = np.array([[2.95309709, 2.51327412], [3.14159265, 4.08407045],
                                        [4.08407045, 2.26194671], [4.46106157, 2.38761042]])

        aqua_globals.random_seed = self.random_seed
        backend = BasicAer.get_backend('qasm_simulator')
        num_qubits = 2
        feature_map = SecondOrderExpansion(feature_dimension=num_qubits, depth=2, entangler_map=[[0, 1]])
        svm = QSVM(feature_map, self.training_data, self.testing_data, None)
        quantum_instance = QuantumInstance(backend, shots=self.shots, seed_simulator=self.random_seed,
                                           seed_transpiler=self.random_seed)

        result = svm.run(quantum_instance)
        np.testing.assert_array_almost_equal(
            result['kernel_matrix_training'], ref_kernel_training, decimal=1)
        np.testing.assert_array_almost_equal(
            result['kernel_matrix_testing'], ref_kernel_testing, decimal=1)

        self.assertEqual(len(result['svm']['support_vectors']), 4)
        np.testing.assert_array_almost_equal(
            result['svm']['support_vectors'], ref_support_vectors, decimal=4)

        np.testing.assert_array_almost_equal(result['svm']['alphas'], ref_alpha, decimal=8)
        np.testing.assert_array_almost_equal(result['svm']['bias'], ref_bias, decimal=8)

        self.assertEqual(result['testing_accuracy'], 0.5)

    def test_qsvm_binary_directly_statevector(self):

        ref_kernel_testing = np. array([[0.1443953, 0.18170069, 0.47479649, 0.14691763],
                                        [0.33041779, 0.37663733, 0.02115561, 0.16106199]])

        ref_support_vectors = np.array([[2.95309709, 2.51327412], [3.14159265, 4.08407045],
                                        [4.08407045, 2.26194671], [4.46106157, 2.38761042]])

        backend = BasicAer.get_backend('statevector_simulator')
        num_qubits = 2
        feature_map = SecondOrderExpansion(feature_dimension=num_qubits, depth=2, entangler_map=[[0, 1]])
        svm = QSVM(feature_map, self.training_data, self.testing_data, None)
        aqua_globals.random_seed = self.random_seed

        quantum_instance = QuantumInstance(backend, seed_transpiler=self.random_seed)
        result = svm.run(quantum_instance)

        ori_alphas = result['svm']['alphas']

        np.testing.assert_array_almost_equal(
            result['kernel_matrix_testing'], ref_kernel_testing, decimal=4)

        self.assertEqual(len(result['svm']['support_vectors']), 4)
        np.testing.assert_array_almost_equal(
            result['svm']['support_vectors'], ref_support_vectors, decimal=4)

        self.assertEqual(result['testing_accuracy'], 0.5)

        file_path = self._get_resource_path('qsvm_test.npz')
        svm.save_model(file_path)

        self.assertTrue(os.path.exists(file_path))

        loaded_svm = QSVM(feature_map)
        loaded_svm.load_model(file_path)

        np.testing.assert_array_almost_equal(
            loaded_svm.ret['svm']['support_vectors'], ref_support_vectors, decimal=4)

        np.testing.assert_array_almost_equal(
            loaded_svm.ret['svm']['alphas'], ori_alphas, decimal=4)

        loaded_test_acc = loaded_svm.test(svm.test_dataset[0], svm.test_dataset[1], quantum_instance)
        self.assertEqual(result['testing_accuracy'], loaded_test_acc)

        np.testing.assert_array_almost_equal(
            loaded_svm.ret['kernel_matrix_testing'], ref_kernel_testing, decimal=4)

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

    def test_qsvm_setup_data(self):

        ref_kernel_testing = np. array([[0.1443953, 0.18170069, 0.47479649, 0.14691763],
                                        [0.33041779, 0.37663733, 0.02115561, 0.16106199]])

        ref_support_vectors = np.array([[2.95309709, 2.51327412], [3.14159265, 4.08407045],
                                        [4.08407045, 2.26194671], [4.46106157, 2.38761042]])

        backend = BasicAer.get_backend('statevector_simulator')
        num_qubits = 2
        feature_map = SecondOrderExpansion(feature_dimension=num_qubits, depth=2, entangler_map=[[0, 1]])

        svm = QSVM(feature_map)

        svm.setup_training_data(self.training_data)
        svm.setup_test_data(self.testing_data)

        aqua_globals.random_seed = self.random_seed

        quantum_instance = QuantumInstance(backend, seed_transpiler=self.random_seed)
        result = svm.run(quantum_instance)

        np.testing.assert_array_almost_equal(
            result['kernel_matrix_testing'], ref_kernel_testing, decimal=4)

        self.assertEqual(len(result['svm']['support_vectors']), 4)
        np.testing.assert_array_almost_equal(
            result['svm']['support_vectors'], ref_support_vectors, decimal=4)

        self.assertEqual(result['testing_accuracy'], 0.5)

    def test_qsvm_multiclass_one_against_all(self):

        backend = BasicAer.get_backend('qasm_simulator')
        training_input = {'A': np.asarray([[0.6560706, 0.17605998], [0.25776033, 0.47628296],
                                           [0.8690704, 0.70847635]]),
                          'B': np.asarray([[0.38857596, -0.33775802], [0.49946978, -0.48727951],
                                           [0.49156185, -0.3660534]]),
                          'C': np.asarray([[-0.68088231, 0.46824423], [-0.56167659, 0.65270294],
                                           [-0.82139073, 0.29941512]])}

        test_input = {'A': np.asarray([[0.57483139, 0.47120732], [0.48372348, 0.25438544],
                                       [0.48142649, 0.15931707]]),
                      'B': np.asarray([[-0.06048935, -0.48345293], [-0.01065613, -0.33910828],
                                       [0.06183066, -0.53376975]]),
                      'C': np.asarray([[-0.74561108, 0.27047295], [-0.69942965, 0.11885162],
                                       [-0.66489165, 0.1181712]])}

        total_array = np.concatenate((test_input['A'], test_input['B'], test_input['C']))

        params = {
            'problem': {'name': 'classification', 'random_seed': self.random_seed},
            'algorithm': {
                'name': 'QSVM',
            },
            'backend': {'shots': self.shots},
            'multiclass_extension': {'name': 'OneAgainstRest'},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2, 'entangler_map': [[0, 1]]}
        }

        algo_input = ClassificationInput(training_input, test_input, total_array)

        result = run_algorithm(params, algo_input, backend=backend)

        expected_accuracy = 0.444444444
        expected_classes = ['A', 'A', 'C', 'A', 'A', 'A', 'A', 'C', 'C']
        self.assertAlmostEqual(result['testing_accuracy'], expected_accuracy, places=4)
        self.assertEqual(result['predicted_classes'], expected_classes)

    def test_qsvm_multiclass_all_pairs(self):

        backend = BasicAer.get_backend('qasm_simulator')
        training_input = {'A': np.asarray([[0.6560706, 0.17605998], [0.25776033, 0.47628296],
                                           [0.8690704, 0.70847635]]),
                          'B': np.asarray([[0.38857596, -0.33775802], [0.49946978, -0.48727951],
                                           [0.49156185, -0.3660534]]),
                          'C': np.asarray([[-0.68088231, 0.46824423], [-0.56167659, 0.65270294],
                                           [-0.82139073, 0.29941512]])}

        test_input = {'A': np.asarray([[0.57483139, 0.47120732], [0.48372348, 0.25438544],
                                       [0.48142649, 0.15931707]]),
                      'B': np.asarray([[-0.06048935, -0.48345293], [-0.01065613, -0.33910828],
                                       [0.06183066, -0.53376975]]),
                      'C': np.asarray([[-0.74561108, 0.27047295], [-0.69942965, 0.11885162],
                                       [-0.66489165, 0.1181712]])}

        total_array = np.concatenate((test_input['A'], test_input['B'], test_input['C']))

        params = {
            'problem': {'name': 'classification', 'random_seed': self.random_seed},
            'algorithm': {
                'name': 'QSVM',
            },
            'backend': {'shots': self.shots},
            'multiclass_extension': {'name': 'AllPairs'},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2, 'entangler_map': [[0, 1]]}
        }

        algo_input = ClassificationInput(training_input, test_input, total_array)
        result = run_algorithm(params, algo_input, backend=backend)
        self.assertAlmostEqual(result['testing_accuracy'], 0.444444444, places=4)
        self.assertEqual(result['predicted_classes'], ['A', 'A', 'C', 'A',
                                                       'A', 'A', 'A', 'C', 'C'])

    def test_qsvm_multiclass_error_correcting_code(self):

        backend = BasicAer.get_backend('qasm_simulator')
        training_input = {'A': np.asarray([[0.6560706, 0.17605998], [0.25776033, 0.47628296],
                                           [0.8690704, 0.70847635]]),
                          'B': np.asarray([[0.38857596, -0.33775802], [0.49946978, -0.48727951],
                                           [0.49156185, -0.3660534]]),
                          'C': np.asarray([[-0.68088231, 0.46824423], [-0.56167659, 0.65270294],
                                           [-0.82139073, 0.29941512]])}

        test_input = {'A': np.asarray([[0.57483139, 0.47120732], [0.48372348, 0.25438544],
                                       [0.48142649, 0.15931707]]),
                      'B': np.asarray([[-0.06048935, -0.48345293], [-0.01065613, -0.33910828],
                                       [0.06183066, -0.53376975]]),
                      'C': np.asarray([[-0.74561108, 0.27047295], [-0.69942965, 0.11885162],
                                       [-0.66489165, 0.1181712]])}

        total_array = np.concatenate((test_input['A'], test_input['B'], test_input['C']))

        params = {
            'problem': {'name': 'classification', 'random_seed': self.random_seed},
            'algorithm': {
                'name': 'QSVM',
            },
            'backend': {'shots': self.shots},
            'multiclass_extension': {'name': 'ErrorCorrectingCode', 'code_size': 5},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2, 'entangler_map': [[0, 1]]}
        }

        algo_input = ClassificationInput(training_input, test_input, total_array)

        result = run_algorithm(params, algo_input, backend=backend)
        self.assertAlmostEqual(result['testing_accuracy'], 0.444444444, places=4)
        self.assertEqual(result['predicted_classes'], ['A', 'A', 'C', 'A',
                                                       'A', 'A', 'A', 'C', 'C'])
