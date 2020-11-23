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

"""Test the VQC algorithm."""

import os
import unittest
import warnings
from test.aqua import QiskitAquaTestCase
from ddt import ddt, data
import numpy as np
from qiskit import BasicAer
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit.aqua import QuantumInstance, aqua_globals, AquaError
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA, COBYLA
from qiskit.ml.circuit.library import RawFeatureVector
from qiskit.aqua.components.feature_maps import RawFeatureVector as LegacyRawFeatureVector
from qiskit.aqua.components.optimizers import L_BFGS_B
from qiskit.ml.datasets import wine, ad_hoc_data


@ddt
class TestVQC(QiskitAquaTestCase):
    """Tests for the VQC algorithm."""

    def setUp(self):
        super().setUp()
        self.seed = 50
        aqua_globals.random_seed = self.seed
        self.training_data = {'A': np.asarray([[2.95309709, 2.51327412], [3.14159265, 4.08407045]]),
                              'B': np.asarray([[4.08407045, 2.26194671], [4.46106157, 2.38761042]])}
        self.testing_data = {'A': np.asarray([[3.83274304, 2.45044227]]),
                             'B': np.asarray([[3.89557489, 0.31415927]])}

        self.ref_opt_params = np.array([0.47352206, -3.75934473, 1.72605939, -4.17669389,
                                        1.28937435, -0.05841719, -0.29853266, -2.04139334,
                                        1.00271775, -1.48133882, -1.18769138, 1.17885493,
                                        7.58873883, -5.27078091, 2.5306601, -4.67393152])

        self.ref_opt_params = np.array([4.40301812e-01, 2.10844304, -2.10118578, -5.25903194,
                                        2.07617769, -9.25865371, -5.33834788, 8.59005180,
                                        3.39886480, 6.33839643, 1.24425033, -1.39701513e+01,
                                        -7.16008545e-03, 3.36206032, 4.38001391, -3.47098082])

        self.ref_train_loss = 0.5869304
        self.ref_prediction_a_probs = [[0.8984375, 0.1015625]]
        self.ref_prediction_a_label = [0]

        self.ryrz_wavefunction = TwoLocal(2, ['ry', 'rz'], 'cz', reps=3, insert_barriers=True)
        self.data_preparation = ZZFeatureMap(2, reps=2)

        self.statevector_simulator = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                                     shots=1,
                                                     seed_simulator=self.seed,
                                                     seed_transpiler=self.seed)
        self.qasm_simulator = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                              shots=1024,
                                              seed_simulator=self.seed,
                                              seed_transpiler=self.seed)

        self.spsa = SPSA(maxiter=10, save_steps=1, c0=4.0, c1=0.1, c2=0.602, c3=0.101,
                         c4=0.0, skip_calibration=True)

    def assertSimpleClassificationIsCorrect(self, vqc, backend=None, ref_opt_params=None,
                                            ref_train_loss=None, ref_test_accuracy=None):
        """Assert running the VQC on the simple data in ``setUp`` works."""
        if backend is None:
            backend = self.qasm_simulator
        if ref_opt_params is None:
            ref_opt_params = self.ref_opt_params
        if ref_train_loss is None:
            ref_train_loss = self.ref_train_loss
        if ref_test_accuracy is None:
            ref_test_accuracy = 0.5

        result = vqc.run(backend)

        with self.subTest(msg='test optimal params'):
            np.testing.assert_array_almost_equal(result['opt_params'], ref_opt_params, decimal=8)

        with self.subTest(msg='test training loss'):
            self.assertAlmostEqual(result['training_loss'], ref_train_loss)

        with self.subTest(msg='check testing accuracy'):
            self.assertEqual(result['testing_accuracy'], ref_test_accuracy)

    def test_basic_aer_qasm(self):
        """Run a basic test case on BasicAer's QASM simulator."""
        data_preparation = self.data_preparation
        wavefunction = self.ryrz_wavefunction
        optimizer = self.spsa
        vqc = VQC(optimizer, data_preparation, wavefunction, self.training_data, self.testing_data)

        self.assertSimpleClassificationIsCorrect(vqc)

    def test_plain_circuits(self):
        """Test running the VQC on QuantumCircuit objects."""
        data_preparation = QuantumCircuit(2).compose(self.data_preparation)
        wavefunction = QuantumCircuit(2).compose(self.ryrz_wavefunction)
        vqc = VQC(self.spsa, data_preparation, wavefunction, self.training_data, self.testing_data)

        self.assertSimpleClassificationIsCorrect(vqc)

    def test_max_evals_grouped(self):
        """Test the VQC with the max_evals_grouped option."""
        data_preparation = self.data_preparation
        wavefunction = self.ryrz_wavefunction

        vqc = VQC(self.spsa, data_preparation, wavefunction, self.training_data, self.testing_data,
                  max_evals_grouped=2)

        self.assertSimpleClassificationIsCorrect(vqc)

    def test_statevector(self):
        """Test running the VQC on BasicAer's Statevector simulator."""
        optimizer = L_BFGS_B(maxfun=200)
        data_preparation = self.data_preparation
        wavefunction = self.ryrz_wavefunction

        vqc = VQC(optimizer, data_preparation, wavefunction, self.training_data, self.testing_data)
        result = vqc.run(self.statevector_simulator)

        with self.subTest(msg='check training loss'):
            self.assertLess(result['training_loss'], 0.12)

        with self.subTest(msg='check testing accuracy'):
            self.assertEqual(result['testing_accuracy'], 0.5)

    def test_minibatching_gradient_free(self):
        """Test the minibatching option with a gradient-free optimizer."""
        n_dim = 2  # dimension of each data point
        _, training_input, test_input, _ = ad_hoc_data(training_size=6,
                                                       test_size=3,
                                                       n=n_dim,
                                                       gap=0.3,
                                                       plot_data=False)
        optimizer = COBYLA(maxiter=40)
        data_preparation = self.data_preparation
        wavefunction = self.ryrz_wavefunction

        vqc = VQC(optimizer, data_preparation, wavefunction, training_input, test_input,
                  minibatch_size=2)
        result = vqc.run(self.qasm_simulator)

        self.log.debug(result['testing_accuracy'])
        self.assertAlmostEqual(result['testing_accuracy'], 0.3333333333333333)

    def test_minibatching_gradient_based(self):
        """Test the minibatching option with a gradient-based optimizer."""
        n_dim = 2  # dimension of each data point
        _, training_input, test_input, _ = ad_hoc_data(training_size=4,
                                                       test_size=2,
                                                       n=n_dim,
                                                       gap=0.3,
                                                       plot_data=False)
        optimizer = L_BFGS_B(maxfun=30)
        data_preparation = self.data_preparation
        wavefunction = TwoLocal(2, ['ry', 'rz'], 'cz', reps=1, insert_barriers=True)

        vqc = VQC(optimizer, data_preparation, wavefunction, training_input, test_input,
                  minibatch_size=2)
        result = vqc.run(self.statevector_simulator)

        self.log.debug(result['testing_accuracy'])
        self.assertAlmostEqual(result['testing_accuracy'], 0.75, places=3)

    def test_save_and_load_model(self):
        """Test saving and loading a model with the VQC."""
        data_preparation = self.data_preparation
        wavefunction = self.ryrz_wavefunction

        vqc = VQC(self.spsa, data_preparation, wavefunction, self.training_data, self.testing_data)
        result = vqc.run(self.qasm_simulator)

        with self.subTest(msg='check optimal params, training loss and testing accuracy'):
            np.testing.assert_array_almost_equal(result['opt_params'], self.ref_opt_params,
                                                 decimal=4)
            np.testing.assert_array_almost_equal(result['training_loss'], self.ref_train_loss,
                                                 decimal=8)
            self.assertEqual(0.5, result['testing_accuracy'])

        file_path = self.get_resource_path('vqc_test.npz')
        vqc.save_model(file_path)

        with self.subTest(msg='assert saved file exists'):
            self.assertTrue(os.path.exists(file_path))

        loaded_vqc = VQC(self.spsa, data_preparation, wavefunction, self.training_data, None)
        loaded_vqc.load_model(file_path)
        loaded_test_acc = loaded_vqc.test(vqc.test_dataset[0],
                                          vqc.test_dataset[1],
                                          self.qasm_simulator)

        with self.subTest(msg='check optimal parameters and testing accuracy of loaded model'):
            np.testing.assert_array_almost_equal(loaded_vqc.ret['opt_params'], self.ref_opt_params,
                                                 decimal=4)
            self.assertEqual(result['testing_accuracy'], loaded_test_acc)

        predicted_probs, predicted_labels = loaded_vqc.predict(self.testing_data['A'],
                                                               self.qasm_simulator)

        with self.subTest(msg='check probs and labels of predicted labels'):
            np.testing.assert_array_almost_equal(predicted_probs, self.ref_prediction_a_probs,
                                                 decimal=8)
            np.testing.assert_array_equal(predicted_labels, self.ref_prediction_a_label)

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:  # pylint: disable=broad-except
                pass

    def test_callback(self):
        """Test the callback function of the VQC."""
        history = {'eval_count': [], 'parameters': [], 'cost': [], 'batch_index': []}

        def store_intermediate_result(eval_count, parameters, cost, batch_index):
            history['eval_count'].append(eval_count)
            history['parameters'].append(parameters)
            history['cost'].append(cost)
            history['batch_index'].append(batch_index)

        optimizer = COBYLA(maxiter=3)
        data_preparation = self.data_preparation
        wavefunction = self.ryrz_wavefunction

        # set up algorithm
        vqc = VQC(optimizer, data_preparation, wavefunction, self.training_data, self.testing_data,
                  callback=store_intermediate_result)

        vqc.run(self.qasm_simulator)

        with self.subTest('eval count'):
            self.assertTrue(all(isinstance(count, int) for count in history['eval_count']))
        with self.subTest('cost'):
            self.assertTrue(all(isinstance(cost, float) for cost in history['cost']))
        with self.subTest('batch index'):
            self.assertTrue(all(isinstance(index, int) for index in history['batch_index']))
        for params in history['parameters']:
            with self.subTest('params'):
                self.assertTrue(all(isinstance(param, float) for param in params))

    def test_same_parameter_names_raises(self):
        """Test that the varform and feature map can have parameters with the same name."""
        aqua_globals.random_seed = self.seed
        var_form = QuantumCircuit(1)
        var_form.ry(Parameter('a'), 0)
        feature_map = QuantumCircuit(1)
        feature_map.rz(Parameter('a'), 0)
        optimizer = SPSA()
        vqc = VQC(optimizer, feature_map, var_form, self.training_data, self.testing_data)

        with self.assertRaises(AquaError):
            _ = vqc.run(BasicAer.get_backend('statevector_simulator'))

    def test_feature_map_without_parameters_warns(self):
        """Test that specifying a feature map with 0 parameters raises a warning."""
        aqua_globals.random_seed = self.seed
        var_form = QuantumCircuit(1)
        var_form.ry(Parameter('a'), 0)
        feature_map = QuantumCircuit(1)
        optimizer = SPSA()
        with self.assertWarns(UserWarning):
            _ = VQC(optimizer, feature_map, var_form, self.training_data, self.testing_data)

    def test_wine(self):
        """Test VQE on the wine dataset."""
        feature_dim = 4  # dimension of each data point
        training_dataset_size = 6
        testing_dataset_size = 3

        _, training_input, test_input, _ = wine(training_size=training_dataset_size,
                                                test_size=testing_dataset_size,
                                                n=feature_dim,
                                                plot_data=False)
        aqua_globals.random_seed = self.seed
        data_preparation = ZZFeatureMap(feature_dim)
        wavefunction = TwoLocal(feature_dim, ['ry', 'rz'], 'cz', reps=2)

        vqc = VQC(COBYLA(maxiter=100), data_preparation, wavefunction, training_input, test_input)
        result = vqc.run(self.statevector_simulator)

        self.log.debug(result['testing_accuracy'])
        self.assertGreater(result['testing_accuracy'], 0.3)

    @data('circuit', 'component')
    def test_raw_feature_vector_on_wine(self, mode):
        """Test VQE on the wine dataset using the ``RawFeatureVector`` as data preparation."""
        feature_dim = 4  # dimension of each data point
        training_dataset_size = 8
        testing_dataset_size = 4

        _, training_input, test_input, _ = wine(training_size=training_dataset_size,
                                                test_size=testing_dataset_size,
                                                n=feature_dim,
                                                plot_data=False)
        if mode == 'component':
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            feature_map = LegacyRawFeatureVector(feature_dimension=feature_dim)
        else:
            feature_map = RawFeatureVector(feature_dimension=feature_dim)

        vqc = VQC(COBYLA(maxiter=100),
                  feature_map,
                  TwoLocal(feature_map.num_qubits, ['ry', 'rz'], 'cz', reps=3),
                  training_input,
                  test_input)
        result = vqc.run(self.statevector_simulator)
        if mode == 'component':
            warnings.filterwarnings('always', category=DeprecationWarning)

        self.log.debug(result['testing_accuracy'])
        self.assertGreater(result['testing_accuracy'], 0.7)


if __name__ == '__main__':
    unittest.main()
