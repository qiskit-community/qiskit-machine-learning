# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
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
from test.aqua import QiskitAquaTestCase

import numpy as np
from ddt import ddt, data

from qiskit import BasicAer, QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua import QuantumInstance, aqua_globals, MissingOptionalLibraryError
from qiskit.aqua.components.multiclass_extensions import (ErrorCorrectingCode,
                                                          AllPairs,
                                                          OneAgainstRest)
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.utils import split_dataset_to_data_and_labels, optimize_svm
from qiskit.ml.datasets import ad_hoc_data


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

        self.ref_kernel_training = np.array([[1., 0.85366667, 0.12341667, 0.36408333],
                                             [0.85366667, 1., 0.11141667, 0.45491667],
                                             [0.12341667, 0.11141667, 1., 0.667],
                                             [0.36408333, 0.45491667, 0.667, 1.]])
        self.ref_kernel_testing = {
            'qasm': np.array([[0.14316667, 0.18208333, 0.4785, 0.14441667],
                              [0.33608333, 0.3765, 0.02316667, 0.15858333]]),
            'statevector': np.array([[0.1443953, 0.18170069, 0.47479649, 0.14691763],
                                     [0.33041779, 0.37663733, 0.02115561, 0.16106199]])
        }
        self.ref_support_vectors = np.array([[2.95309709, 2.51327412], [3.14159265, 4.08407045],
                                             [4.08407045, 2.26194671], [4.46106157, 2.38761042]])
        self.ref_alpha = np.array([0.34902907, 1.48325913, 0.03073616, 1.80155205])
        self.ref_bias = np.array([-0.03059395])

        self.qasm_simulator = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                              shots=self.shots,
                                              seed_simulator=self.random_seed,
                                              seed_transpiler=self.random_seed)
        self.statevector_simulator = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                                     shots=1,
                                                     seed_simulator=self.random_seed,
                                                     seed_transpiler=self.random_seed)

        self.data_preparation = ZZFeatureMap(feature_dimension=2, reps=2)

    @data('library', 'circuit')
    def test_binary(self, mode):
        """Test QSVM on binary classification on BasicAer's QASM simulator."""
        if mode == 'circuit':
            data_preparation = QuantumCircuit(2).compose(self.data_preparation)
        else:
            data_preparation = self.data_preparation

        svm = QSVM(data_preparation, self.training_data, self.testing_data, None, lambda2=0)

        try:
            result = svm.run(self.qasm_simulator)
            np.testing.assert_array_almost_equal(result['kernel_matrix_training'],
                                                 self.ref_kernel_training, decimal=1)
            np.testing.assert_array_almost_equal(result['kernel_matrix_testing'],
                                                 self.ref_kernel_testing['qasm'], decimal=1)

            self.assertEqual(len(result['svm']['support_vectors']), 4)
            np.testing.assert_array_almost_equal(result['svm']['support_vectors'],
                                                 self.ref_support_vectors, decimal=4)

            np.testing.assert_array_almost_equal(result['svm']['alphas'], self.ref_alpha, decimal=8)
            np.testing.assert_array_almost_equal(result['svm']['bias'], self.ref_bias, decimal=8)

            self.assertEqual(result['testing_accuracy'], 0.5)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    def test_binary_directly_statevector(self):
        """Test QSVM on binary classification on BasicAer's statevector simulator.

        Also tests saving and loading models."""
        data_preparation = self.data_preparation
        svm = QSVM(data_preparation, self.training_data, self.testing_data, None, lambda2=0)

        file_path = self.get_resource_path('qsvm_test.npz')
        try:
            result = svm.run(self.statevector_simulator)

            ori_alphas = result['svm']['alphas']

            np.testing.assert_array_almost_equal(result['kernel_matrix_testing'],
                                                 self.ref_kernel_testing['statevector'], decimal=4)

            self.assertEqual(len(result['svm']['support_vectors']), 4)
            np.testing.assert_array_almost_equal(result['svm']['support_vectors'],
                                                 self.ref_support_vectors, decimal=4)

            self.assertEqual(result['testing_accuracy'], 0.5)

            svm.save_model(file_path)

            self.assertTrue(os.path.exists(file_path))

            loaded_svm = QSVM(data_preparation)
            loaded_svm.load_model(file_path)

            np.testing.assert_array_almost_equal(loaded_svm.ret['svm']['support_vectors'],
                                                 self.ref_support_vectors, decimal=4)

            np.testing.assert_array_almost_equal(loaded_svm.ret['svm']['alphas'], ori_alphas,
                                                 decimal=4)

            loaded_test_acc = loaded_svm.test(svm.test_dataset[0],
                                              svm.test_dataset[1],
                                              self.statevector_simulator)
            self.assertEqual(result['testing_accuracy'], loaded_test_acc)

            np.testing.assert_array_almost_equal(loaded_svm.ret['kernel_matrix_testing'],
                                                 self.ref_kernel_testing['statevector'], decimal=4)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:  # pylint: disable=broad-except
                    pass

    def test_setup_data(self):
        """Test the setup_*_data methods of QSVM."""
        data_preparation = self.data_preparation
        try:
            svm = QSVM(data_preparation)

            svm.setup_training_data(self.training_data)
            svm.setup_test_data(self.testing_data)
            result = svm.run(self.statevector_simulator)

            np.testing.assert_array_almost_equal(result['kernel_matrix_testing'],
                                                 self.ref_kernel_testing['statevector'], decimal=4)

            self.assertEqual(len(result['svm']['support_vectors']), 4)
            np.testing.assert_array_almost_equal(result['svm']['support_vectors'],
                                                 self.ref_support_vectors, decimal=4)

            self.assertEqual(result['testing_accuracy'], 0.5)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    @data('one_vs_all', 'all_vs_all', 'error_correcting')
    def test_multiclass(self, multiclass_extension):
        """ QSVM Multiclass One Against All test """
        train_input = {'A': np.asarray([[0.6560706, 0.17605998],
                                        [0.25776033, 0.47628296],
                                        [0.8690704, 0.70847635]]),
                       'B': np.asarray([[0.38857596, -0.33775802],
                                        [0.49946978, -0.48727951],
                                        [0.49156185, -0.3660534]]),
                       'C': np.asarray([[-0.68088231, 0.46824423],
                                        [-0.56167659, 0.65270294],
                                        [-0.82139073, 0.29941512]])}

        test_input = {'A': np.asarray([[0.57483139, 0.47120732],
                                       [0.48372348, 0.25438544],
                                       [0.48142649, 0.15931707]]),
                      'B': np.asarray([[-0.06048935, -0.48345293],
                                       [-0.01065613, -0.33910828],
                                       [0.06183066, -0.53376975]]),
                      'C': np.asarray([[-0.74561108, 0.27047295],
                                       [-0.69942965, 0.11885162],
                                       [-0.66489165, 0.1181712]])}

        method = {'one_vs_all': OneAgainstRest(),
                  'all_vs_all': AllPairs(),
                  'error_correcting': ErrorCorrectingCode(code_size=5)}

        accuracy = {'one_vs_all': 0.444444444,
                    'all_vs_all': 0.444444444,
                    'error_correcting': 0.555555555}

        predicted_classes = {
            'one_vs_all': ['A', 'A', 'C', 'A', 'A', 'A', 'A', 'C', 'C'],
            'all_vs_all': ['A', 'A', 'C', 'A', 'A', 'A', 'A', 'C', 'C'],
            'error_correcting': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'C']
        }

        total_array = np.concatenate((test_input['A'], test_input['B'], test_input['C']))

        data_preparation = self.data_preparation
        try:
            svm = QSVM(data_preparation, train_input, test_input, total_array,
                       multiclass_extension=method[multiclass_extension], lambda2=0)
            result = svm.run(self.qasm_simulator)
            self.assertAlmostEqual(result['testing_accuracy'], accuracy[multiclass_extension],
                                   places=4)
            self.assertEqual(result['predicted_classes'], predicted_classes[multiclass_extension])
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    def test_matrix_psd(self):
        """ Test kernel matrix positive semi-definite enforcement. """
        try:
            from cvxpy.error import DQCPError
        except ImportError:
            self.skipTest('cvxpy does not appeat to be installed')

        seed = 10598
        feature_dim = 2
        _, training_input, _, _ = ad_hoc_data(
            training_size=10,
            test_size=5,
            n=feature_dim,
            gap=0.3
        )
        training_input, _ = split_dataset_to_data_and_labels(training_input)
        training_data = training_input[0]
        training_labels = training_input[1]
        labels = training_labels * 2 - 1  # map label from 0 --> -1 and 1 --> 1
        labels = labels.astype(float)

        feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement='linear')

        try:
            with self.assertRaises(DQCPError):
                # Sampling noise means that the kernel matrix will not quite be positive
                # semi-definite which will cause the optimize svm to fail
                backend = BasicAer.get_backend('qasm_simulator')
                quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed,
                                                   seed_transpiler=seed)
                kernel_matrix = QSVM.get_kernel_matrix(quantum_instance, feature_map=feature_map,
                                                       x1_vec=training_data, enforce_psd=False)
                _ = optimize_svm(kernel_matrix, labels, lambda2=0)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))
            return

        # This time we enforce that the matrix be positive semi-definite which runs logic to
        # make it so.
        backend = BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed,
                                           seed_transpiler=seed)
        kernel_matrix = QSVM.get_kernel_matrix(quantum_instance, feature_map=feature_map,
                                               x1_vec=training_data, enforce_psd=True)
        alpha, b, support = optimize_svm(kernel_matrix, labels, lambda2=0)

        expected_alpha = [0.855861781, 2.59807482, 0, 0.962959215,
                          1.08141696, 0.217172547, 0, 0,
                          0.786462904, 0, 0.969727949, 1.98066946,
                          0, 0, 1.62049430, 0,
                          0.394212728, 0, 0.507740935, 1.02910286]
        expected_b = [-0.17543365]
        expected_support = [True, True, False, True, True, True, False, False, True, False,
                            True, True, False, False, True, False, True, False, True, True]
        np.testing.assert_array_almost_equal(alpha, expected_alpha)
        np.testing.assert_array_almost_equal(b, expected_b)
        np.testing.assert_array_equal(support, expected_support)
