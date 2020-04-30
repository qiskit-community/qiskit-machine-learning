# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test QSVM """

import os
import warnings
from test.aqua import QiskitAquaTestCase
import numpy as np
from ddt import ddt, data
from qiskit import BasicAer, QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.components.multiclass_extensions import (ErrorCorrectingCode,
                                                          AllPairs,
                                                          OneAgainstRest)
from qiskit.aqua.algorithms import QSVM


@ddt
class TestQSVM(QiskitAquaTestCase):
    """ Test QSVM """

    def setUp(self):
        super().setUp()
        self.random_seed = 10598
        self.shots = 12000
        aqua_globals.random_seed = self.random_seed
        self.training_data = {'A': np.asarray([[2.95309709, 2.51327412],
                                               [3.14159265, 4.08407045]]),
                              'B': np.asarray([[4.08407045, 2.26194671],
                                               [4.46106157, 2.38761042]])}
        self.testing_data = {'A': np.asarray([[3.83274304, 2.45044227]]),
                             'B': np.asarray([[3.89557489, 0.31415927]])}

        num_qubits = 2

        warnings.filterwarnings('ignore', category=DeprecationWarning)
        # data encoding using a FeatureMap type
        feature_map = SecondOrderExpansion(feature_dimension=num_qubits,
                                           depth=2,
                                           entangler_map=[[0, 1]])
        warnings.filterwarnings('always', category=DeprecationWarning)

        # data encoding using a circuit library object
        library_circuit = ZZFeatureMap(feature_dimension=num_qubits, reps=2)

        # data encoding using a plain QuantumCircuit
        circuit = QuantumCircuit(num_qubits).compose(library_circuit)
        circuit.ordered_parameters = library_circuit.ordered_parameters

        self.data_preparation = {'wrapped': feature_map,
                                 'circuit': circuit,
                                 'library': library_circuit}

    @data('wrapped', 'circuit', 'library')
    def test_qsvm_binary(self, data_preparation_type):
        """ QSVM Binary test """
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

        backend = BasicAer.get_backend('qasm_simulator')
        data_preparation = self.data_preparation[data_preparation_type]
        if data_preparation_type == 'wrapped':
            warnings.filterwarnings('ignore', category=DeprecationWarning)
        svm = QSVM(data_preparation, self.training_data, self.testing_data, None)
        if data_preparation_type == 'wrapped':
            warnings.filterwarnings('always', category=DeprecationWarning)
        quantum_instance = QuantumInstance(backend,
                                           shots=self.shots,
                                           seed_simulator=self.random_seed,
                                           seed_transpiler=self.random_seed)

        try:
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
        except NameError as ex:
            self.skipTest(str(ex))

    @data('wrapped', 'circuit', 'library')
    def test_qsvm_binary_directly_statevector(self, data_preparation_type):
        """ QSVM Binary Directly Statevector test """
        ref_kernel_testing = np. array([[0.1443953, 0.18170069, 0.47479649, 0.14691763],
                                        [0.33041779, 0.37663733, 0.02115561, 0.16106199]])

        ref_support_vectors = np.array([[2.95309709, 2.51327412], [3.14159265, 4.08407045],
                                        [4.08407045, 2.26194671], [4.46106157, 2.38761042]])

        backend = BasicAer.get_backend('statevector_simulator')
        data_preparation = self.data_preparation[data_preparation_type]
        if data_preparation_type == 'wrapped':
            warnings.filterwarnings('ignore', category=DeprecationWarning)
        svm = QSVM(data_preparation, self.training_data, self.testing_data, None)
        if data_preparation_type == 'wrapped':
            warnings.filterwarnings('always', category=DeprecationWarning)
        quantum_instance = QuantumInstance(backend, seed_transpiler=self.random_seed,
                                           seed_simulator=self.random_seed)

        file_path = self.get_resource_path('qsvm_test.npz')
        try:
            result = svm.run(quantum_instance)

            ori_alphas = result['svm']['alphas']

            np.testing.assert_array_almost_equal(
                result['kernel_matrix_testing'], ref_kernel_testing, decimal=4)

            self.assertEqual(len(result['svm']['support_vectors']), 4)
            np.testing.assert_array_almost_equal(
                result['svm']['support_vectors'], ref_support_vectors, decimal=4)

            self.assertEqual(result['testing_accuracy'], 0.5)

            svm.save_model(file_path)

            self.assertTrue(os.path.exists(file_path))

            loaded_svm = QSVM(feature_map)
            loaded_svm.load_model(file_path)

            np.testing.assert_array_almost_equal(
                loaded_svm.ret['svm']['support_vectors'], ref_support_vectors, decimal=4)

            np.testing.assert_array_almost_equal(
                loaded_svm.ret['svm']['alphas'], ori_alphas, decimal=4)

            loaded_test_acc = loaded_svm.test(svm.test_dataset[0],
                                              svm.test_dataset[1],
                                              quantum_instance)
            self.assertEqual(result['testing_accuracy'], loaded_test_acc)

            np.testing.assert_array_almost_equal(
                loaded_svm.ret['kernel_matrix_testing'], ref_kernel_testing, decimal=4)
        except NameError as ex:
            self.skipTest(str(ex))
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:  # pylint: disable=broad-except
                    pass

    @data('wrapped', 'circuit', 'library')
    def test_qsvm_setup_data(self, data_preparation_type):
        """ QSVM Setup Data test """
        ref_kernel_testing = np. array([[0.1443953, 0.18170069, 0.47479649, 0.14691763],
                                        [0.33041779, 0.37663733, 0.02115561, 0.16106199]])

        ref_support_vectors = np.array([[2.95309709, 2.51327412], [3.14159265, 4.08407045],
                                        [4.08407045, 2.26194671], [4.46106157, 2.38761042]])

        backend = BasicAer.get_backend('statevector_simulator')

        data_preparation = self.data_preparation[data_preparation_type]
        try:
            if data_preparation_type == 'wrapped':
                warnings.filterwarnings('ignore', category=DeprecationWarning)
            svm = QSVM(data_preparation)
            if data_preparation_type == 'wrapped':
                warnings.filterwarnings('always', category=DeprecationWarning)

            svm.setup_training_data(self.training_data)
            svm.setup_test_data(self.testing_data)
            quantum_instance = QuantumInstance(backend, seed_transpiler=self.random_seed,
                                               seed_simulator=self.random_seed)
            result = svm.run(quantum_instance)

            np.testing.assert_array_almost_equal(
                result['kernel_matrix_testing'], ref_kernel_testing, decimal=4)

            self.assertEqual(len(result['svm']['support_vectors']), 4)
            np.testing.assert_array_almost_equal(
                result['svm']['support_vectors'], ref_support_vectors, decimal=4)

            self.assertEqual(result['testing_accuracy'], 0.5)
        except NameError as ex:
            self.skipTest(str(ex))

    @data('wrapped', 'circuit', 'library')
    def test_qsvm_multiclass_one_against_all(self, data_preparation_type):
        """ QSVM Multiclass One Against All test """
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

        aqua_globals.random_seed = self.random_seed
        data_preparation = self.data_preparation[data_preparation_type]
        try:
            if data_preparation_type == 'wrapped':
                warnings.filterwarnings('ignore', category=DeprecationWarning)
            svm = QSVM(data_preparation, training_input, test_input, total_array,
                       multiclass_extension=OneAgainstRest())
            if data_preparation_type == 'wrapped':
                warnings.filterwarnings('always', category=DeprecationWarning)
            quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                               shots=self.shots,
                                               seed_simulator=aqua_globals.random_seed,
                                               seed_transpiler=aqua_globals.random_seed)
            result = svm.run(quantum_instance)
            expected_accuracy = 0.444444444
            expected_classes = ['A', 'A', 'C', 'A', 'A', 'A', 'A', 'C', 'C']
            self.assertAlmostEqual(result['testing_accuracy'], expected_accuracy, places=4)
            self.assertEqual(result['predicted_classes'], expected_classes)
        except NameError as ex:
            self.skipTest(str(ex))

    @data('wrapped', 'circuit', 'library')
    def test_qsvm_multiclass_all_pairs(self, data_preparation_type):
        """ QSVM Multiclass All Pairs test """
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

        aqua_globals.random_seed = self.random_seed
        data_preparation = self.data_preparation[data_preparation_type]
        try:
            if data_preparation_type == 'wrapped':
                warnings.filterwarnings('ignore', category=DeprecationWarning)
            svm = QSVM(data_preparation, training_input, test_input, total_array,
                       multiclass_extension=AllPairs())
            if data_preparation_type == 'wrapped':
                warnings.filterwarnings('always', category=DeprecationWarning)

            quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                               shots=self.shots,
                                               seed_simulator=aqua_globals.random_seed,
                                               seed_transpiler=aqua_globals.random_seed)
            result = svm.run(quantum_instance)
            self.assertAlmostEqual(result['testing_accuracy'], 0.444444444, places=4)
            self.assertEqual(result['predicted_classes'], ['A', 'A', 'C', 'A',
                                                           'A', 'A', 'A', 'C', 'C'])
        except NameError as ex:
            self.skipTest(str(ex))

    @data('wrapped', 'circuit', 'library')
    def test_qsvm_multiclass_error_correcting_code(self, data_preparation_type):
        """ QSVM Multiclass error Correcting Code test """
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

        aqua_globals.random_seed = self.random_seed
        data_preparation = self.data_preparation[data_preparation_type]
        try:
            if data_preparation_type == 'wrapped':
                warnings.filterwarnings('ignore', category=DeprecationWarning)
            svm = QSVM(data_preparation, training_input, test_input, total_array,
                       multiclass_extension=ErrorCorrectingCode(code_size=5))
            if data_preparation_type == 'wrapped':
                warnings.filterwarnings('always', category=DeprecationWarning)

            quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                               shots=self.shots,
                                               seed_simulator=aqua_globals.random_seed,
                                               seed_transpiler=aqua_globals.random_seed)
            result = svm.run(quantum_instance)
            self.assertAlmostEqual(result['testing_accuracy'], 0.444444444, places=4)
            self.assertEqual(result['predicted_classes'], ['A', 'A', 'C', 'A',
                                                           'A', 'A', 'A', 'C', 'C'])
        except NameError as ex:
            self.skipTest(str(ex))
