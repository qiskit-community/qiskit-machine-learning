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

""" Test VQC """

import os
import unittest
from test.aqua import QiskitAquaTestCase
import numpy as np
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA, COBYLA
from qiskit.aqua.components.feature_maps import SecondOrderExpansion, RawFeatureVector
from qiskit.aqua.components.variational_forms import RYRZ, RY
from qiskit.aqua.components.optimizers import L_BFGS_B
from qiskit.aqua.utils import get_feature_dimension
from qiskit.ml.datasets import wine, ad_hoc_data


class TestVQC(QiskitAquaTestCase):
    """ Test VQC """
    def setUp(self):
        super().setUp()
        self.seed = 1376
        aqua_globals.random_seed = self.seed
        self.training_data = {'A': np.asarray([[2.95309709, 2.51327412], [3.14159265, 4.08407045]]),
                              'B': np.asarray([[4.08407045, 2.26194671], [4.46106157, 2.38761042]])}
        self.testing_data = {'A': np.asarray([[3.83274304, 2.45044227]]),
                             'B': np.asarray([[3.89557489, 0.31415927]])}

        self.ref_opt_params = np.array([10.03814083, -12.22048954, -7.58026833, -2.42392954,
                                        12.91555293, 13.44064652, -2.89951454, -10.20639406,
                                        0.81414546, -1.00551752, -4.7988307, 14.00831419,
                                        8.26008064, -7.07543736, 11.43368677, -5.74857438])
        self.ref_train_loss = 0.69366523
        self.ref_prediction_a_probs = [[0.79882812, 0.20117188]]
        self.ref_prediction_a_label = [0]

    def test_vqc(self):
        """ vqc test """
        aqua_globals.random_seed = self.seed
        optimizer = SPSA(max_trials=10, save_steps=1,
                         c0=4.0, c1=0.1, c2=0.602, c3=0.101, c4=0.0, skip_calibration=True)
        feature_map = SecondOrderExpansion(
            feature_dimension=get_feature_dimension(self.training_data), depth=2)
        var_form = RYRZ(num_qubits=feature_map.num_qubits, depth=3)
        vqc = VQC(optimizer, feature_map, var_form, self.training_data, self.testing_data)
        quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                           shots=1024,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = vqc.run(quantum_instance)

        np.testing.assert_array_almost_equal(result['opt_params'],
                                             self.ref_opt_params, decimal=8)
        np.testing.assert_array_almost_equal(result['training_loss'],
                                             self.ref_train_loss, decimal=8)

        self.assertEqual(1.0, result['testing_accuracy'])

    def test_vqc_with_max_evals_grouped(self):
        """ vqc with max evals grouped test """
        aqua_globals.random_seed = self.seed
        optimizer = SPSA(max_trials=10, save_steps=1,
                         c0=4.0, c1=0.1, c2=0.602, c3=0.101, c4=0.0, skip_calibration=True)
        feature_map = SecondOrderExpansion(
            feature_dimension=get_feature_dimension(self.training_data), depth=2)
        var_form = RYRZ(num_qubits=feature_map.num_qubits, depth=3)
        vqc = VQC(optimizer, feature_map, var_form, self.training_data, self.testing_data,
                  max_evals_grouped=2)
        quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                           shots=1024,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = vqc.run(quantum_instance)
        np.testing.assert_array_almost_equal(result['opt_params'],
                                             self.ref_opt_params, decimal=8)
        np.testing.assert_array_almost_equal(result['training_loss'],
                                             self.ref_train_loss, decimal=8)

        self.assertEqual(1.0, result['testing_accuracy'])

    def test_vqc_statevector(self):
        """ vqc statevector test """
        aqua_globals.random_seed = 10598
        optimizer = COBYLA()
        feature_map = SecondOrderExpansion(
            feature_dimension=get_feature_dimension(self.training_data), depth=2)
        var_form = RYRZ(num_qubits=feature_map.num_qubits, depth=3)
        vqc = VQC(optimizer, feature_map, var_form, self.training_data, self.testing_data)
        quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = vqc.run(quantum_instance)
        ref_train_loss = 0.1059404
        np.testing.assert_array_almost_equal(result['training_loss'], ref_train_loss, decimal=4)

        self.assertEqual(result['testing_accuracy'], 0.5)

    # we use the ad_hoc dataset (see the end of this file) to test the accuracy.
    def test_vqc_minibatching_no_gradient_support(self):
        """ vqc minibatching with no gradient support test """
        n_dim = 2  # dimension of each data point
        seed = 1024
        aqua_globals.random_seed = seed
        _, training_input, test_input, _ = ad_hoc_data(training_size=6,
                                                       test_size=3,
                                                       n=n_dim,
                                                       gap=0.3,
                                                       plot_data=False)
        backend = BasicAer.get_backend('statevector_simulator')
        num_qubits = n_dim
        optimizer = COBYLA(maxiter=40)
        feature_map = SecondOrderExpansion(feature_dimension=num_qubits, depth=2)
        var_form = RYRZ(num_qubits=num_qubits, depth=3)
        vqc = VQC(optimizer, feature_map, var_form, training_input, test_input, minibatch_size=2)
        quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed,
                                           optimization_level=0)
        result = vqc.run(quantum_instance)
        self.log.debug(result['testing_accuracy'])
        self.assertGreaterEqual(result['testing_accuracy'], 0.5)

    def test_vqc_minibatching_with_gradient_support(self):
        """ vqc minibatching with gradient support test """
        n_dim = 2  # dimension of each data point
        seed = 1024
        aqua_globals.random_seed = seed
        _, training_input, test_input, _ = ad_hoc_data(training_size=6,
                                                       test_size=2,
                                                       n=n_dim,
                                                       gap=0.3,
                                                       plot_data=False)
        backend = BasicAer.get_backend('statevector_simulator')
        num_qubits = n_dim
        optimizer = L_BFGS_B(maxfun=30)
        feature_map = SecondOrderExpansion(feature_dimension=num_qubits, depth=2)
        var_form = RYRZ(num_qubits=num_qubits, depth=1)
        vqc = VQC(optimizer, feature_map, var_form, training_input, test_input, minibatch_size=2)
        quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)
        result = vqc.run(quantum_instance)
        self.log.debug(result['testing_accuracy'])
        self.assertGreaterEqual(result['testing_accuracy'], 0.5)

    def test_save_and_load_model(self):
        """ save and load model test """
        aqua_globals.random_seed = self.seed
        backend = BasicAer.get_backend('qasm_simulator')

        num_qubits = 2
        optimizer = SPSA(max_trials=10, save_steps=1, c0=4.0, skip_calibration=True)
        feature_map = SecondOrderExpansion(feature_dimension=num_qubits, depth=2)
        var_form = RYRZ(num_qubits=num_qubits, depth=3)

        vqc = VQC(optimizer, feature_map, var_form, self.training_data, self.testing_data)
        quantum_instance = QuantumInstance(backend,
                                           shots=1024,
                                           seed_simulator=self.seed,
                                           seed_transpiler=self.seed)
        result = vqc.run(quantum_instance)

        np.testing.assert_array_almost_equal(result['opt_params'],
                                             self.ref_opt_params, decimal=4)
        np.testing.assert_array_almost_equal(result['training_loss'],
                                             self.ref_train_loss, decimal=8)

        self.assertEqual(1.0, result['testing_accuracy'])

        file_path = self.get_resource_path('vqc_test.npz')
        vqc.save_model(file_path)

        self.assertTrue(os.path.exists(file_path))

        loaded_vqc = VQC(optimizer, feature_map, var_form, self.training_data, None)
        loaded_vqc.load_model(file_path)

        np.testing.assert_array_almost_equal(
            loaded_vqc.ret['opt_params'], self.ref_opt_params, decimal=4)

        loaded_test_acc = loaded_vqc.test(vqc.test_dataset[0],
                                          vqc.test_dataset[1],
                                          quantum_instance)
        self.assertEqual(result['testing_accuracy'], loaded_test_acc)

        predicted_probs, predicted_labels = loaded_vqc.predict(self.testing_data['A'],
                                                               quantum_instance)
        np.testing.assert_array_almost_equal(predicted_probs,
                                             self.ref_prediction_a_probs,
                                             decimal=8)
        np.testing.assert_array_equal(predicted_labels, self.ref_prediction_a_label)

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:  # pylint: disable=broad-except
                pass

    def test_vqc_callback(self):
        """ vqc callback test """
        tmp_filename = 'qvqc_callback_test.csv'
        is_file_exist = os.path.exists(self.get_resource_path(tmp_filename))
        if is_file_exist:
            os.remove(self.get_resource_path(tmp_filename))

        def store_intermediate_result(eval_count, parameters, cost, batch_index):
            with open(self.get_resource_path(tmp_filename), 'a') as file:
                content = "{},{},{:.5f},{}".format(eval_count, parameters, cost, batch_index)
                print(content, file=file, flush=True)

        aqua_globals.random_seed = self.seed
        backend = BasicAer.get_backend('qasm_simulator')

        num_qubits = 2
        optimizer = COBYLA(maxiter=3)
        feature_map = SecondOrderExpansion(feature_dimension=num_qubits, depth=2)
        var_form = RY(num_qubits=num_qubits, depth=1)

        vqc = VQC(optimizer, feature_map, var_form, self.training_data,
                  self.testing_data, callback=store_intermediate_result)
        quantum_instance = QuantumInstance(backend,
                                           shots=1024,
                                           seed_simulator=self.seed,
                                           seed_transpiler=self.seed)
        vqc.run(quantum_instance)

        is_file_exist = os.path.exists(self.get_resource_path(tmp_filename))
        self.assertTrue(is_file_exist, "Does not store content successfully.")

        # check the content
        ref_content = [
            ['0', '[-0.58205563 -2.97987177 -0.73153057  1.06577518]', '0.46841', '0'],
            ['1', '[ 0.41794437 -2.97987177 -0.73153057  1.06577518]', '0.31861', '1'],
            ['2', '[ 0.41794437 -1.97987177 -0.73153057  1.06577518]', '0.45975', '2'],
        ]
        try:
            with open(self.get_resource_path(tmp_filename)) as file:
                idx = 0
                for record in file.readlines():
                    eval_count, parameters, cost, batch_index = record.split(",")
                    self.assertEqual(eval_count.strip(), ref_content[idx][0])
                    self.assertEqual(parameters, ref_content[idx][1])
                    self.assertEqual(cost.strip(), ref_content[idx][2])
                    self.assertEqual(batch_index.strip(), ref_content[idx][3])
                    idx += 1
        finally:
            if is_file_exist:
                os.remove(self.get_resource_path(tmp_filename))

    def test_vqc_on_wine(self):
        """ vqc on wine test """
        feature_dim = 4  # dimension of each data point
        training_dataset_size = 6
        testing_dataset_size = 3

        _, training_input, test_input, _ = wine(training_size=training_dataset_size,
                                                test_size=testing_dataset_size,
                                                n=feature_dim,
                                                plot_data=False)
        aqua_globals.random_seed = self.seed
        feature_map = SecondOrderExpansion(feature_dimension=feature_dim)
        vqc = VQC(COBYLA(maxiter=100),
                  feature_map,
                  RYRZ(feature_map.num_qubits, depth=1),
                  training_input,
                  test_input)
        result = vqc.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         shots=1024,
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))
        self.log.debug(result['testing_accuracy'])

        self.assertLess(result['testing_accuracy'], 0.6)

    def test_vqc_with_raw_feature_vector_on_wine(self):
        """ vqc with raw features vector on wine test """
        feature_dim = 4  # dimension of each data point
        training_dataset_size = 8
        testing_dataset_size = 4

        _, training_input, test_input, _ = wine(training_size=training_dataset_size,
                                                test_size=testing_dataset_size,
                                                n=feature_dim,
                                                plot_data=False)
        aqua_globals.random_seed = self.seed
        feature_map = RawFeatureVector(feature_dimension=feature_dim)
        vqc = VQC(COBYLA(maxiter=100),
                  feature_map,
                  RYRZ(feature_map.num_qubits, depth=3),
                  training_input,
                  test_input)
        result = vqc.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         shots=1024,
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))
        self.log.debug(result['testing_accuracy'])

        self.assertGreater(result['testing_accuracy'], 0.8)


if __name__ == '__main__':
    unittest.main()
