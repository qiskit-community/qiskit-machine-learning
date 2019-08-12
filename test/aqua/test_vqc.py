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
import unittest

import numpy as np
import scipy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from test.aqua.common import QiskitAquaTestCase
from qiskit import BasicAer
from qiskit.aqua.input import ClassificationInput
from qiskit.aqua import run_algorithm, QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA, COBYLA
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.components.variational_forms import RYRZ, RY
from qiskit.aqua.components.optimizers import L_BFGS_B


class TestVQC(QiskitAquaTestCase):

    def setUp(self):
        super().setUp()
        aqua_globals.random_seed = self.random_seed = 1376
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

        self.vqc_input = ClassificationInput(self.training_data, self.testing_data)

    def test_vqc_via_run_algorithm(self):
        params = {
            'problem': {'name': 'classification', 'random_seed': self.random_seed},
            'algorithm': {'name': 'VQC'},
            'backend': {'provider': 'qiskit.BasicAer', 'name': 'qasm_simulator', 'shots': 1024},
            'optimizer': {'name': 'SPSA', 'max_trials': 10, 'save_steps': 1},
            'variational_form': {'name': 'RYRZ', 'depth': 3},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2}
        }
        result = run_algorithm(params, self.vqc_input)

        np.testing.assert_array_almost_equal(result['opt_params'], self.ref_opt_params, decimal=8)
        np.testing.assert_array_almost_equal(result['training_loss'], self.ref_train_loss, decimal=8)

        self.assertEqual(1.0, result['testing_accuracy'])

    def test_vqc_with_max_evals_grouped(self):
        params = {
            'problem': {'name': 'classification', 'random_seed': self.random_seed},
            'algorithm': {'name': 'VQC', 'max_evals_grouped': 2},
            'backend': {'provider': 'qiskit.BasicAer', 'name': 'qasm_simulator', 'shots': 1024},
            'optimizer': {'name': 'SPSA', 'max_trials': 10, 'save_steps': 1},
            'variational_form': {'name': 'RYRZ', 'depth': 3},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2}
        }
        result = run_algorithm(params, self.vqc_input)

        np.testing.assert_array_almost_equal(result['opt_params'], self.ref_opt_params, decimal=8)
        np.testing.assert_array_almost_equal(result['training_loss'], self.ref_train_loss, decimal=8)

        self.assertEqual(1.0, result['testing_accuracy'])

    def test_vqc_statevector_via_run_algorithm(self):
        # TODO: cache only work with optimization_level 0
        params = {
            'problem': {'name': 'classification',
                        'random_seed': 10598,
                        'circuit_optimization_level': 0,
                        'circuit_caching': True,
                        'skip_qobj_deepcopy': True,
                        'skip_qobj_validation': True,
                        'circuit_cache_file': None,
                        },
            'algorithm': {'name': 'VQC'},
            'backend': {'provider': 'qiskit.BasicAer', 'name': 'statevector_simulator'},
            'optimizer': {'name': 'COBYLA'},
            'variational_form': {'name': 'RYRZ', 'depth': 3},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2}
        }
        result = run_algorithm(params, self.vqc_input)
        ref_train_loss = 0.1059404
        np.testing.assert_array_almost_equal(result['training_loss'], ref_train_loss, decimal=4)

        self.assertEqual(result['testing_accuracy'], 0.5)

    # we use the ad_hoc dataset (see the end of this file) to test the accuracy.
    def test_vqc_minibatching_no_gradient_support(self):
        n_dim = 2  # dimension of each data point
        seed = 1024
        np.random.seed(seed)
        sample_Total, training_input, test_input, class_labels = ad_hoc_data(training_size=8,
                                                                             test_size=4,
                                                                             n=n_dim, gap=0.3)
        aqua_globals.random_seed = seed
        backend = BasicAer.get_backend('statevector_simulator')
        num_qubits = n_dim
        optimizer = COBYLA()
        feature_map = SecondOrderExpansion(feature_dimension=num_qubits, depth=2)
        var_form = RYRZ(num_qubits=num_qubits, depth=3)
        vqc = VQC(optimizer, feature_map, var_form, training_input, test_input, minibatch_size=2)
        quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)
        result = vqc.run(quantum_instance)
        vqc_accuracy_threshold = 0.8
        self.log.debug(result['testing_accuracy'])
        self.assertGreater(result['testing_accuracy'], vqc_accuracy_threshold)

    def test_vqc_minibatching_with_gradient_support(self):
        n_dim = 2  # dimension of each data point
        seed = 1024
        np.random.seed(seed)
        sample_Total, training_input, test_input, class_labels = ad_hoc_data(training_size=8,
                                                                             test_size=4,
                                                                             n=n_dim, gap=0.3)
        aqua_globals.random_seed = seed
        backend = BasicAer.get_backend('statevector_simulator')
        num_qubits = n_dim
        optimizer = L_BFGS_B(maxfun=100)
        feature_map = SecondOrderExpansion(feature_dimension=num_qubits, depth=2)
        var_form = RYRZ(num_qubits=num_qubits, depth=2)
        vqc = VQC(optimizer, feature_map, var_form, training_input, test_input, minibatch_size=2)
        # TODO: cache only work with optimization_level 0
        quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed, optimization_level=0)
        result = vqc.run(quantum_instance)
        vqc_accuracy_threshold = 0.8
        self.log.debug(result['testing_accuracy'])
        self.assertGreater(result['testing_accuracy'], vqc_accuracy_threshold)

    def test_save_and_load_model(self):
        np.random.seed(self.random_seed)

        aqua_globals.random_seed = self.random_seed
        backend = BasicAer.get_backend('qasm_simulator')

        num_qubits = 2
        optimizer = SPSA(max_trials=10, save_steps=1, c0=4.0, skip_calibration=True)
        feature_map = SecondOrderExpansion(feature_dimension=num_qubits, depth=2)
        var_form = RYRZ(num_qubits=num_qubits, depth=3)

        vqc = VQC(optimizer, feature_map, var_form, self.training_data, self.testing_data)
        quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=self.random_seed, seed_transpiler=self.random_seed)
        result = vqc.run(quantum_instance)

        np.testing.assert_array_almost_equal(result['opt_params'], self.ref_opt_params, decimal=4)
        np.testing.assert_array_almost_equal(result['training_loss'], self.ref_train_loss, decimal=8)

        self.assertEqual(1.0, result['testing_accuracy'])

        file_path = self._get_resource_path('vqc_test.npz')
        vqc.save_model(file_path)

        self.assertTrue(os.path.exists(file_path))

        loaded_vqc = VQC(optimizer, feature_map, var_form, self.training_data, None)
        loaded_vqc.load_model(file_path)

        np.testing.assert_array_almost_equal(
            loaded_vqc.ret['opt_params'], self.ref_opt_params, decimal=4)

        loaded_test_acc = loaded_vqc.test(vqc.test_dataset[0], vqc.test_dataset[1], quantum_instance)
        self.assertEqual(result['testing_accuracy'], loaded_test_acc)

        predicted_probs, predicted_labels = loaded_vqc.predict(self.testing_data['A'], quantum_instance)
        np.testing.assert_array_almost_equal(predicted_probs, self.ref_prediction_a_probs, decimal=8)
        np.testing.assert_array_equal(predicted_labels, self.ref_prediction_a_label)
        if quantum_instance.has_circuit_caching:
            self.assertLess(quantum_instance._circuit_cache.misses, 3)

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

    def test_vqc_callback(self):

        tmp_filename = 'qvqc_callback_test.csv'
        is_file_exist = os.path.exists(self._get_resource_path(tmp_filename))
        if is_file_exist:
            os.remove(self._get_resource_path(tmp_filename))

        def store_intermediate_result(eval_count, parameters, cost, batch_index):
            with open(self._get_resource_path(tmp_filename), 'a') as f:
                content = "{},{},{:.5f},{}".format(eval_count, parameters, cost, batch_index)
                print(content, file=f, flush=True)

        np.random.seed(self.random_seed)
        aqua_globals.random_seed = self.random_seed
        backend = BasicAer.get_backend('qasm_simulator')

        num_qubits = 2
        optimizer = COBYLA(maxiter=3)
        feature_map = SecondOrderExpansion(feature_dimension=num_qubits, depth=2)
        var_form = RY(num_qubits=num_qubits, depth=1)

        vqc = VQC(optimizer, feature_map, var_form, self.training_data,
                  self.testing_data, callback=store_intermediate_result)
        quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=self.random_seed, seed_transpiler=self.random_seed)
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
            with open(self._get_resource_path(tmp_filename)) as f:
                idx = 0
                for record in f.readlines():
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
        feature_dim = 4  # dimension of each data point
        training_dataset_size = 8
        testing_dataset_size = 4
        random_seed = 10598
        np.random.seed(random_seed)

        sample_total, training_input, test_input, class_labels = wine_data(
            training_size=training_dataset_size,
            test_size=testing_dataset_size,
            n=feature_dim
        )
        # TODO: cache only work with optimization_level 0
        params = {
            'problem': {'name': 'classification',
                        'random_seed': self.random_seed,
                        'circuit_optimization_level': 0,
                        'circuit_caching': True,
                        'skip_qobj_deepcopy': True,
                        'skip_qobj_validation': True,
                        'circuit_cache_file': None,
                        },
            'algorithm': {'name': 'VQC'},
            'backend': {'provider': 'qiskit.BasicAer', 'name': 'statevector_simulator'},
            'optimizer': {'name': 'COBYLA', 'maxiter': 200},
            'variational_form': {'name': 'RYRZ', 'depth': 3},
        }

        result = run_algorithm(params, ClassificationInput(training_input, test_input))
        self.log.debug(result['testing_accuracy'])

        self.assertLess(result['testing_accuracy'], 0.6)

    def test_vqc_with_raw_feature_vector_on_wine(self):
        feature_dim = 4  # dimension of each data point
        training_dataset_size = 8
        testing_dataset_size = 4
        random_seed = 10598
        np.random.seed(random_seed)

        sample_total, training_input, test_input, class_labels = wine_data(
            training_size=training_dataset_size,
            test_size=testing_dataset_size,
            n=feature_dim
        )
        # TODO: cache only work with optimization_level 0
        params = {
            'problem': {'name': 'classification',
                        'random_seed': self.random_seed,
                        'circuit_optimization_level': 0,
                        'circuit_caching': True,
                        'skip_qobj_deepcopy': True,
                        'skip_qobj_validation': True,
                        'circuit_cache_file': None,
                        },
            'algorithm': {'name': 'VQC'},
            'backend': {'provider': 'qiskit.BasicAer', 'name': 'statevector_simulator'},
            'optimizer': {'name': 'COBYLA', 'maxiter': 200},
            'variational_form': {'name': 'RYRZ', 'depth': 3},
            'feature_map': {'name': 'RawFeatureVector', 'feature_dimension': feature_dim}
        }

        result = run_algorithm(params, ClassificationInput(training_input, test_input))
        self.log.debug(result['testing_accuracy'])

        self.assertGreater(result['testing_accuracy'], 0.8)


def wine_data(training_size, test_size, n):
    class_labels = [r'A', r'B', r'C']

    data, target = datasets.load_wine(True)
    sample_train, sample_test, label_train, label_test = train_test_split(
        data, target, test_size=test_size, random_state=7
    )

    # Now we standarize for gaussian around 0 with unit variance
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


def ad_hoc_data(training_size, test_size, n, gap):
    class_labels = [r'A', r'B']
    if n == 2:
        N = 100
    elif n == 3:
        N = 20   # courseness of data seperation

    label_train = np.zeros(2*(training_size+test_size))
    sample_train = []
    sampleA = [[0 for x in range(n)] for y in range(training_size+test_size)]
    sampleB = [[0 for x in range(n)] for y in range(training_size+test_size)]

    sample_Total = [[[0 for x in range(N)] for y in range(N)] for z in range(N)]

    # interactions = np.transpose(np.array([[1, 0], [0, 1], [1, 1]]))

    steps = 2 * np.pi / N

    # sx = np.array([[0, 1], [1, 0]])
    # X = np.asmatrix(sx)
    # sy = np.array([[0, -1j], [1j, 0]])
    # Y = np.asmatrix(sy)
    sz = np.array([[1, 0], [0, -1]])
    Z = np.asmatrix(sz)
    J = np.array([[1, 0], [0, 1]])
    J = np.asmatrix(J)
    H = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    H2 = np.kron(H, H)
    H3 = np.kron(H, H2)
    H = np.asmatrix(H)
    H2 = np.asmatrix(H2)
    H3 = np.asmatrix(H3)

    f = np.arange(2 ** n)

    my_array = [[0 for x in range(n)] for y in range(2 ** n)]

    for arindex in range(len(my_array)):
        temp_f = bin(f[arindex])[2:].zfill(n)
        for findex in range(n):
            my_array[arindex][findex] = int(temp_f[findex])

    my_array = np.asarray(my_array)
    my_array = np.transpose(my_array)

    # Define decision functions
    maj = (-1) ** (2 * my_array.sum(axis=0) > n)
    parity = (-1) ** (my_array.sum(axis=0))
    # dict1 = (-1) ** (my_array[0])
    if n == 2:
        D = np.diag(parity)
    elif n == 3:
        D = np.diag(maj)

    Basis = np.random.random((2 ** n, 2 ** n)) + 1j * np.random.random((2 ** n, 2 ** n))
    Basis = np.asmatrix(Basis).getH() * np.asmatrix(Basis)

    [S, U] = np.linalg.eig(Basis)

    idx = S.argsort()[::-1]
    S = S[idx]
    U = U[:, idx]

    M = (np.asmatrix(U)).getH() * np.asmatrix(D) * np.asmatrix(U)

    psi_plus = np.transpose(np.ones(2)) / np.sqrt(2)
    psi_0 = 1
    for k in range(n):
        psi_0 = np.kron(np.asmatrix(psi_0), np.asmatrix(psi_plus))

    sample_total_A = []
    sample_total_B = []
    sample_total_void = []
    if n == 2:
        for n1 in range(N):
            for n2 in range(N):
                x1 = steps * n1
                x2 = steps * n2
                phi = x1 * np.kron(Z, J) + x2 * np.kron(J, Z) + (np.pi-x1) * (np.pi-x2) * np.kron(Z, Z)
                # pylint: disable=no-member
                Uu = scipy.linalg.expm(1j * phi)
                psi = np.asmatrix(Uu) * H2 * np.asmatrix(Uu) * np.transpose(psi_0)
                temp = np.real(psi.getH() * M * psi)
                temp = temp.item()
                if temp > gap:
                    sample_Total[n1][n2] = +1
                elif temp < -gap:
                    sample_Total[n1][n2] = -1
                else:
                    sample_Total[n1][n2] = 0

        # Now sample randomly from sample_Total a number of times training_size+testing_size
        tr = 0
        while tr < (training_size + test_size):
            draw1 = np.random.choice(N)
            draw2 = np.random.choice(N)
            if sample_Total[draw1][draw2] == +1:
                sampleA[tr] = [2 * np.pi * draw1 / N, 2 * np.pi * draw2 / N]
                tr += 1

        tr = 0
        while tr < (training_size+test_size):
            draw1 = np.random.choice(N)
            draw2 = np.random.choice(N)
            if sample_Total[draw1][draw2] == -1:
                sampleB[tr] = [2 * np.pi * draw1 / N, 2 * np.pi * draw2 / N]
                tr += 1

        sample_train = [sampleA, sampleB]

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
        for n1 in range(N):
            for n2 in range(N):
                for n3 in range(N):
                    x1 = steps * n1
                    x2 = steps * n2
                    x3 = steps * n3
                    phi = x1 * np.kron(np.kron(Z, J), J) + \
                        x2 * np.kron(np.kron(J, Z), J) + \
                        x3 * np.kron(np.kron(J, J), Z) + \
                        (np.pi - x1) * (np.pi - x2) * np.kron(np.kron(Z, Z), J) + \
                        (np.pi - x2) * (np.pi - x3) * np.kron(np.kron(J, Z), Z) + \
                        (np.pi - x1) * (np.pi - x3) * np.kron(np.kron(Z, J), Z)
                    # pylint: disable=no-member
                    Uu = scipy.linalg.expm(1j * phi)
                    psi = np.asmatrix(Uu) * H3 * np.asmatrix(Uu) * np.transpose(psi_0)
                    temp = np.real(psi.getH() * M * psi)
                    temp = temp.item()
                    if temp > gap:
                        sample_Total[n1][n2][n3] = +1
                        sample_total_A.append([n1, n2, n3])
                    elif temp < -gap:
                        sample_Total[n1][n2][n3] = -1
                        sample_total_B.append([n1, n2, n3])
                    else:
                        sample_Total[n1][n2][n3] = 0
                        sample_total_void.append([n1, n2, n3])

        # Now sample randomly from sample_Total a number of times training_size+testing_size
        tr = 0
        while tr < (training_size + test_size):
            draw1 = np.random.choice(N)
            draw2 = np.random.choice(N)
            draw3 = np.random.choice(N)
            if sample_Total[draw1][draw2][draw3] == +1:
                sampleA[tr] = [2 * np.pi * draw1 / N, 2 * np.pi * draw2 / N, 2 * np.pi * draw3 / N]
                tr += 1

        tr = 0
        while tr < (training_size + test_size):
            draw1 = np.random.choice(N)
            draw2 = np.random.choice(N)
            draw3 = np.random.choice(N)
            if sample_Total[draw1][draw2][draw3] == -1:
                sampleB[tr] = [2 * np.pi * draw1 / N, 2 * np.pi * draw2 / N, 2 * np.pi * draw3 / N]
                tr += 1

        sample_train = [sampleA, sampleB]

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

    return sample_Total, training_input, test_input, class_labels


if __name__ == '__main__':
    unittest.main()
