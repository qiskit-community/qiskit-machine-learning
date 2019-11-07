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

""" Test VQC """

import os
import unittest
from test.aqua.common import QiskitAquaTestCase
import warnings
import numpy as np
import scipy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from qiskit import BasicAer
from qiskit.aqua.input import ClassificationInput
from qiskit.aqua import run_algorithm, QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA, COBYLA
from qiskit.aqua.components.feature_maps import SecondOrderExpansion, RawFeatureVector
from qiskit.aqua.components.variational_forms import RYRZ, RY
from qiskit.aqua.components.optimizers import L_BFGS_B
from qiskit.aqua.utils import get_feature_dimension


class TestVQC(QiskitAquaTestCase):
    """ Test VQC """
    def setUp(self):
        super().setUp()
        warnings.filterwarnings("ignore", message=aqua_globals.CONFIG_DEPRECATION_MSG, category=DeprecationWarning)
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

    def test_vqc_via_run_algorithm(self):
        """ vqc via run algorithm test """
        params = {
            'problem': {'name': 'classification', 'random_seed': self.seed},
            'algorithm': {'name': 'VQC'},
            'backend': {'provider': 'qiskit.BasicAer', 'name': 'qasm_simulator', 'shots': 1024},
            'optimizer': {'name': 'SPSA', 'max_trials': 10, 'save_steps': 1},
            'variational_form': {'name': 'RYRZ', 'depth': 3},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2}
        }
        result = run_algorithm(params, ClassificationInput(self.training_data, self.testing_data))

        np.testing.assert_array_almost_equal(result['opt_params'],
                                             self.ref_opt_params, decimal=8)
        np.testing.assert_array_almost_equal(result['training_loss'],
                                             self.ref_train_loss, decimal=8)

        self.assertEqual(1.0, result['testing_accuracy'])

    def test_vqc_with_max_evals_grouped(self):
        """ vqc with max evals grouped test """
        aqua_globals.random_seed = self.seed
        feature_map = SecondOrderExpansion(get_feature_dimension(self.training_data), depth=2)
        vqc = VQC(SPSA(max_trials=10, save_steps=1),
                  feature_map,
                  RYRZ(feature_map.num_qubits, depth=3),
                  self.training_data,
                  self.testing_data,
                  max_evals_grouped=2)
        result = vqc.run(QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                         shots=1024))

        np.testing.assert_array_almost_equal(result['opt_params'],
                                             self.ref_opt_params, decimal=8)
        np.testing.assert_array_almost_equal(result['training_loss'],
                                             self.ref_train_loss, decimal=8)

        self.assertEqual(1.0, result['testing_accuracy'])

    def test_vqc_statevector_via_algorithm(self):
        """ vqc statevector via algorithm test """
        aqua_globals.random_seed = 10598
        feature_map = SecondOrderExpansion(get_feature_dimension(self.training_data), depth=2)
        vqc = VQC(COBYLA(),
                  feature_map,
                  RYRZ(feature_map.num_qubits, depth=3),
                  self.training_data,
                  self.testing_data,
                  max_evals_grouped=2)
        result = vqc.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         shots=1024))
        ref_train_loss = 0.1059404
        np.testing.assert_array_almost_equal(result['training_loss'], ref_train_loss, decimal=4)

        self.assertEqual(result['testing_accuracy'], 0.5)

    # we use the ad_hoc dataset (see the end of this file) to test the accuracy.
    def test_vqc_minibatching_no_gradient_support(self):
        """ vqc minibatching with no gradient support test """
        n_dim = 2  # dimension of each data point
        seed = 1024
        aqua_globals.random_seed = seed
        _, training_input, test_input, _ = _ad_hoc_data(training_size=6,
                                                        test_size=3,
                                                        n=n_dim, gap=0.3)
        backend = BasicAer.get_backend('statevector_simulator')
        num_qubits = n_dim
        optimizer = COBYLA(maxiter=40)
        feature_map = SecondOrderExpansion(feature_dimension=num_qubits, depth=2)
        var_form = RYRZ(num_qubits=num_qubits, depth=3)
        vqc = VQC(optimizer, feature_map, var_form, training_input, test_input, minibatch_size=2)
        quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed,
                                           optimization_level=0)
        result = vqc.run(quantum_instance)
        vqc_accuracy = 0.666
        self.log.debug(result['testing_accuracy'])
        self.assertGreaterEqual(result['testing_accuracy'], vqc_accuracy)

    def test_vqc_minibatching_with_gradient_support(self):
        """ vqc minibatching with gradient support test """
        n_dim = 2  # dimension of each data point
        seed = 1024
        aqua_globals.random_seed = seed
        _, training_input, test_input, _ = _ad_hoc_data(training_size=4,
                                                        test_size=2,
                                                        n=n_dim, gap=0.3)
        backend = BasicAer.get_backend('statevector_simulator')
        num_qubits = n_dim
        optimizer = L_BFGS_B(maxfun=30)
        feature_map = SecondOrderExpansion(feature_dimension=num_qubits, depth=2)
        var_form = RYRZ(num_qubits=num_qubits, depth=1)
        vqc = VQC(optimizer, feature_map, var_form, training_input, test_input, minibatch_size=2)
        quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)
        result = vqc.run(quantum_instance)
        vqc_accuracy = 0.5
        self.log.debug(result['testing_accuracy'])
        self.assertAlmostEqual(result['testing_accuracy'], vqc_accuracy, places=3)

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

        file_path = self._get_resource_path('vqc_test.npz')
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
        is_file_exist = os.path.exists(self._get_resource_path(tmp_filename))
        if is_file_exist:
            os.remove(self._get_resource_path(tmp_filename))

        def store_intermediate_result(eval_count, parameters, cost, batch_index):
            with open(self._get_resource_path(tmp_filename), 'a') as file:
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

        is_file_exist = os.path.exists(self._get_resource_path(tmp_filename))
        self.assertTrue(is_file_exist, "Does not store content successfully.")

        # check the content
        ref_content = [
            ['0', '[-0.58205563 -2.97987177 -0.73153057  1.06577518]', '0.46841', '0'],
            ['1', '[ 0.41794437 -2.97987177 -0.73153057  1.06577518]', '0.31861', '1'],
            ['2', '[ 0.41794437 -1.97987177 -0.73153057  1.06577518]', '0.45975', '2'],
        ]
        try:
            with open(self._get_resource_path(tmp_filename)) as file:
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
                os.remove(self._get_resource_path(tmp_filename))

    def test_vqc_on_wine(self):
        """ vqc on wine test """
        feature_dim = 4  # dimension of each data point
        training_dataset_size = 6
        testing_dataset_size = 3

        _, training_input, test_input, _ = _wine_data(
            training_size=training_dataset_size,
            test_size=testing_dataset_size,
            n=feature_dim
        )
        aqua_globals.random_seed = self.seed
        feature_map = SecondOrderExpansion(feature_dimension=feature_dim)
        vqc = VQC(COBYLA(maxiter=100),
                  feature_map,
                  RYRZ(feature_map.num_qubits, depth=1),
                  training_input,
                  test_input)
        result = vqc.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         shots=1024))
        self.log.debug(result['testing_accuracy'])

        self.assertLess(result['testing_accuracy'], 0.6)

    def test_vqc_with_raw_feature_vector_on_wine(self):
        """ vqc with raw features vector on wine test """
        feature_dim = 4  # dimension of each data point
        training_dataset_size = 8
        testing_dataset_size = 4

        _, training_input, test_input, _ = _wine_data(
            training_size=training_dataset_size,
            test_size=testing_dataset_size,
            n=feature_dim
        )
        aqua_globals.random_seed = self.seed
        feature_map = RawFeatureVector(feature_dimension=feature_dim)
        vqc = VQC(COBYLA(maxiter=100),
                  feature_map,
                  RYRZ(feature_map.num_qubits, depth=3),
                  training_input,
                  test_input)
        result = vqc.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         shots=1024))
        self.log.debug(result['testing_accuracy'])

        self.assertGreater(result['testing_accuracy'], 0.8)


def _wine_data(training_size, test_size, n):
    class_labels = [r'A', r'B', r'C']

    data, target = datasets.load_wine(True)
    sample_train, sample_test, label_train, _ = train_test_split(
        data, target, test_size=test_size, random_state=7
    )

    # Now we standardize for gaussian around 0 with unit variance
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)

    # Now reduce number of features to number of qubits
    pca = PCA(n_components=n).fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)

    # Scale to the range (-1,+1)
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)
    # Pick training size number of samples from each distro
    training_input = {
        key: (sample_train[label_train == k, :])[:training_size]
        for k, key in enumerate(class_labels)
    }
    test_input = {
        key: (sample_train[label_train == k, :])[training_size:(training_size + test_size)]
        for k, key in enumerate(class_labels)
    }
    return sample_train, training_input, test_input, class_labels


def _ad_hoc_data(training_size, test_size, n, gap):
    class_labels = [r'A', r'B']
    n_v = 0
    if n == 2:
        n_v = 100
    elif n == 3:
        n_v = 20   # coarseness of data separation

    label_train = np.zeros(2*(training_size+test_size))
    sample_train = []
    sample_a = [[0 for x in range(n)] for y in range(training_size+test_size)]
    sample_b = [[0 for x in range(n)] for y in range(training_size+test_size)]

    sample_total = [[[0 for x in range(n_v)] for y in range(n_v)] for z in range(n_v)]

    # interactions = np.transpose(np.array([[1, 0], [0, 1], [1, 1]]))

    steps = 2 * np.pi / n_v

    # sx = np.array([[0, 1], [1, 0]])
    # X = np.asmatrix(sx)
    # sy = np.array([[0, -1j], [1j, 0]])
    # Y = np.asmatrix(sy)
    s_z = np.array([[1, 0], [0, -1]])
    z_m = np.asmatrix(s_z)
    j_m = np.array([[1, 0], [0, 1]])
    j_m = np.asmatrix(j_m)
    h_v = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    h_2 = np.kron(h_v, h_v)
    h_3 = np.kron(h_v, h_2)
    h_v = np.asmatrix(h_v)
    h_2 = np.asmatrix(h_2)
    h_3 = np.asmatrix(h_3)

    f_r = np.arange(2 ** n)

    my_array = [[0 for x in range(n)] for y in range(2 ** n)]

    for arindex, my_value in enumerate(my_array):
        temp_f = bin(f_r[arindex])[2:].zfill(n)
        for findex in range(n):
            my_value[findex] = int(temp_f[findex])

    my_array = np.asarray(my_array)
    my_array = np.transpose(my_array)

    # Define decision functions
    maj = (-1) ** (2 * my_array.sum(axis=0) > n)
    parity = (-1) ** (my_array.sum(axis=0))
    # dict1 = (-1) ** (my_array[0])
    d_v = None
    if n == 2:
        d_v = np.diag(parity)
    elif n == 3:
        d_v = np.diag(maj)

    basis = aqua_globals.random.random_sample((2 ** n, 2 ** n)) + \
        1j * aqua_globals.random.random_sample((2 ** n, 2 ** n))
    basis = np.asmatrix(basis).getH() * np.asmatrix(basis)

    [s_v, u_v] = np.linalg.eig(basis)

    idx = s_v.argsort()[::-1]
    s_v = s_v[idx]
    u_v = u_v[:, idx]

    m_v = (np.asmatrix(u_v)).getH() * np.asmatrix(d_v) * np.asmatrix(u_v)

    psi_plus = np.transpose(np.ones(2)) / np.sqrt(2)
    psi_0 = 1
    for _ in range(n):
        psi_0 = np.kron(np.asmatrix(psi_0), np.asmatrix(psi_plus))

    sample_total_a = []
    sample_total_b = []
    sample_total_void = []
    if n == 2:
        for n_1 in range(n_v):
            for n_2 in range(n_v):
                x_1 = steps * n_1
                x_2 = steps * n_2
                phi = x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m) + \
                    (np.pi-x_1) * (np.pi-x_2) * np.kron(z_m, z_m)
                # pylint: disable=no-member
                u_u = scipy.linalg.expm(1j * phi)
                psi = np.asmatrix(u_u) * h_2 * np.asmatrix(u_u) * np.transpose(psi_0)
                temp = np.real(psi.getH() * m_v * psi)
                temp = temp.item()
                if temp > gap:
                    sample_total[n_1][n_2] = +1
                elif temp < -gap:
                    sample_total[n_1][n_2] = -1
                else:
                    sample_total[n_1][n_2] = 0

        # Now sample randomly from sample_total a number of times training_size+testing_size
        t_r = 0
        while t_r < (training_size + test_size):
            draw1 = aqua_globals.random.choice(n_v)
            draw2 = aqua_globals.random.choice(n_v)
            if sample_total[draw1][draw2] == +1:
                sample_a[t_r] = [2 * np.pi * draw1 / n_v, 2 * np.pi * draw2 / n_v]
                t_r += 1

        t_r = 0
        while t_r < (training_size+test_size):
            draw1 = aqua_globals.random.choice(n_v)
            draw2 = aqua_globals.random.choice(n_v)
            if sample_total[draw1][draw2] == -1:
                sample_b[t_r] = [2 * np.pi * draw1 / n_v, 2 * np.pi * draw2 / n_v]
                t_r += 1

        sample_train = [sample_a, sample_b]

        for lindex in range(training_size + test_size):
            label_train[lindex] = 0
        for lindex in range(training_size + test_size):
            label_train[training_size + test_size + lindex] = 1
        label_train = label_train.astype(int)
        sample_train = np.reshape(sample_train, (2 * (training_size + test_size), n))
        training_input = {
            key: (sample_train[label_train == k, :])[:training_size]
            for k, key in enumerate(class_labels)
        }
        test_input = {
            key: (sample_train[label_train == k, :])[training_size:(training_size + test_size)]
            for k, key in enumerate(class_labels)
        }

    elif n == 3:
        for n_1 in range(n_v):
            for n_2 in range(n_v):
                for n_3 in range(n_v):
                    x_1 = steps * n_1
                    x_2 = steps * n_2
                    x_3 = steps * n_3
                    phi = x_1 * np.kron(np.kron(z_m, j_m), j_m) + \
                        x_2 * np.kron(np.kron(j_m, z_m), j_m) + \
                        x_3 * np.kron(np.kron(j_m, j_m), z_m) + \
                        (np.pi - x_1) * (np.pi - x_2) * np.kron(np.kron(z_m, z_m), j_m) + \
                        (np.pi - x_2) * (np.pi - x_3) * np.kron(np.kron(j_m, z_m), z_m) + \
                        (np.pi - x_1) * (np.pi - x_3) * np.kron(np.kron(z_m, j_m), z_m)
                    # pylint: disable=no-member
                    u_u = scipy.linalg.expm(1j * phi)
                    psi = np.asmatrix(u_u) * h_3 * np.asmatrix(u_u) * np.transpose(psi_0)
                    temp = np.real(psi.getH() * m_v * psi)
                    temp = temp.item()
                    if temp > gap:
                        sample_total[n_1][n_2][n_3] = +1
                        sample_total_a.append([n_1, n_2, n_3])
                    elif temp < -gap:
                        sample_total[n_1][n_2][n_3] = -1
                        sample_total_b.append([n_1, n_2, n_3])
                    else:
                        sample_total[n_1][n_2][n_3] = 0
                        sample_total_void.append([n_1, n_2, n_3])

        # Now sample randomly from sample_total a number of times training_size+testing_size
        t_r = 0
        while t_r < (training_size + test_size):
            draw1 = aqua_globals.random.choice(n_v)
            draw2 = aqua_globals.random.choice(n_v)
            draw3 = aqua_globals.random.choice(n_v)
            if sample_total[draw1][draw2][draw3] == +1:
                sample_a[t_r] = \
                    [2 * np.pi * draw1 / n_v, 2 * np.pi * draw2 / n_v, 2 * np.pi * draw3 / n_v]
                t_r += 1

        t_r = 0
        while t_r < (training_size + test_size):
            draw1 = aqua_globals.random.choice(n_v)
            draw2 = aqua_globals.random.choice(n_v)
            draw3 = aqua_globals.random.choice(n_v)
            if sample_total[draw1][draw2][draw3] == -1:
                sample_b[t_r] = \
                    [2 * np.pi * draw1 / n_v, 2 * np.pi * draw2 / n_v, 2 * np.pi * draw3 / n_v]
                t_r += 1

        sample_train = [sample_a, sample_b]

        for lindex in range(training_size + test_size):
            label_train[lindex] = 0
        for lindex in range(training_size + test_size):
            label_train[training_size + test_size + lindex] = 1
        label_train = label_train.astype(int)
        sample_train = np.reshape(sample_train, (2 * (training_size + test_size), n))
        training_input = {
            key: (sample_train[label_train == k, :])[:training_size]
            for k, key in enumerate(class_labels)
        }
        test_input = {
            key: (sample_train[label_train == k, :])[training_size:(training_size + test_size)]
            for k, key in enumerate(class_labels)
        }

    return sample_total, training_input, test_input, class_labels


if __name__ == '__main__':
    unittest.main()
