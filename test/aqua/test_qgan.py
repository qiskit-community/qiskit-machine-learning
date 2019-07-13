# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# =============================================================================

import unittest

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from qiskit.aqua.components.uncertainty_models import UniformDistribution, UnivariateVariationalDistribution
from qiskit.aqua.components.variational_forms import RY

from qiskit.aqua.algorithms.adaptive.qgan.qgan import QGAN
from qiskit.aqua.input import QGANInput
from qiskit.aqua import aqua_globals, QuantumInstance, run_algorithm
from qiskit.aqua.components.initial_states import Custom

from qiskit import BasicAer

from test.aqua.common import QiskitAquaTestCase


class TestQGAN(QiskitAquaTestCase):

    def setUp(self):
        super().setUp()

        # Number training data samples
        N = 5000
        # Load data samples from log-normal distribution with mean=1 and standard deviation=1
        mu = 1
        sigma = 1
        self._real_data = np.random.lognormal(mean=mu, sigma=sigma, size=N)
        # Set the data resolution
        # Set upper and lower data values as list of k min/max data values [[min_0,max_0],...,[min_k-1,max_k-1]]
        self._bounds = [0., 3.]
        # Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
        num_qubits = [2]
        # Batch size
        batch_size = 100
        # Set number of training epochs
        num_epochs = 10
        self._params_torch = {'algorithm': {'name': 'QGAN',
                                            'num_qubits': num_qubits,
                                            'batch_size': batch_size,
                                            'num_epochs': num_epochs},
                              'problem': {'name': 'distribution_learning_loading', 'random_seed': 7},
                              'generative_network': {'name': 'QuantumGenerator',
                                                     'bounds': self._bounds,
                                                     'num_qubits': num_qubits,
                                                     'init_params': None,
                                                     'snapshot_dir': None
                                                     },
                              'discriminative_network': {'name': 'PytorchDiscriminator',
                                                         'n_features': len(num_qubits)}
                              }
        self._params_numpy = {'algorithm': {'name': 'QGAN',
                                            'num_qubits': num_qubits,
                                            'batch_size': batch_size,
                                            'num_epochs': num_epochs},
                              'problem': {'name': 'distribution_learning_loading', 'random_seed': 7},
                              'generative_network': {'name': 'QuantumGenerator',
                                                     'bounds': self._bounds,
                                                     'num_qubits': num_qubits,
                                                     'init_params': None,
                                                     'snapshot_dir': None
                                                     },
                              'discriminative_network': {'name': 'NumpyDiscriminator',
                                                         'n_features': len(num_qubits)}
                              }

        # Initialize qGAN
        self.qgan = QGAN(self._real_data, self._bounds, num_qubits, batch_size, num_epochs, snapshot_dir=None)
        self.qgan.seed = 7
        # Set quantum instance to run the quantum generator
        self.quantum_instance_statevector = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                                            circuit_caching=False, seed_simulator=2, seed_transpiler=2)
        self.quantum_instance_qasm = QuantumInstance(backend=BasicAer.get_backend('qasm_simulator'), shots=1000,
                                                     circuit_caching=False, seed_simulator=2, seed_transpiler=2)
        # Set entangler map
        entangler_map = [[0, 1]]

        # Set an initial state for the generator circuit
        init_dist = UniformDistribution(sum(num_qubits), low=self._bounds[0], high=self._bounds[1])
        q = QuantumRegister(sum(num_qubits), name='q')
        qc = QuantumCircuit(q)
        init_dist.build(qc, q)
        init_distribution = Custom(num_qubits=sum(num_qubits), circuit=qc)
        # Set variational form
        var_form = RY(sum(num_qubits), depth=1, initial_state=init_distribution, entangler_map=entangler_map,
                      entanglement_gate='cz')
        # Set generator's initial parameters
        init_params = aqua_globals.random.rand(var_form._num_parameters) * 2 * 1e-2
        # Set generator circuit
        g_circuit = UnivariateVariationalDistribution(sum(num_qubits), var_form, init_params,
                                                      low=self._bounds[0],
                                                      high=self._bounds[1])
        # initial_distribution=init_distribution,
        # Set quantum generator
        self.qgan.set_generator(generator_circuit=g_circuit)

    def test_sample_generation(self):
        samples_statevector, weights_statevector = self.qgan._generator.get_output(self.quantum_instance_statevector,
                                                                                   shots=100)
        samples_qasm, weights_qasm = self.qgan._generator.get_output(self.quantum_instance_qasm, shots=100)
        samples_qasm, weights_qasm = zip(*sorted(zip(samples_qasm, weights_qasm)))
        for i, weight_q in enumerate(weights_qasm):
            self.assertAlmostEqual(weight_q, weights_statevector[i], delta=0.1)

    def test_qgan_training(self):
        trained_statevector = self.qgan.run(self.quantum_instance_statevector)
        trained_qasm = self.qgan.run(self.quantum_instance_qasm)
        self.assertAlmostEqual(trained_qasm['rel_entr'], trained_statevector['rel_entr'], delta=0.1)

    def test_qgan_training_run_algo_torch(self):
        try:
            algo_input = QGANInput(self._real_data, self._bounds)
            trained_statevector = run_algorithm(params=self._params_torch, algo_input=algo_input,
                                                backend=BasicAer.get_backend('statevector_simulator'))
            trained_qasm = run_algorithm(self._params_torch, algo_input, backend=BasicAer.get_backend('qasm_simulator'))
            self.assertAlmostEqual(trained_qasm['rel_entr'], trained_statevector['rel_entr'], delta=0.1)
        except Exception as e:
            self.skipTest("Torch may not be installed: '{}'".format(str(e)))

    def test_qgan_training_run_algo_numpy(self):
        algo_input = QGANInput(self._real_data, self._bounds)
        trained_statevector = run_algorithm(params=self._params_numpy, algo_input=algo_input,
                                            backend=BasicAer.get_backend('statevector_simulator'))
        trained_qasm = run_algorithm(self._params_numpy, algo_input, backend=BasicAer.get_backend('qasm_simulator'))
        self.assertAlmostEqual(trained_qasm['rel_entr'], trained_statevector['rel_entr'], delta=0.1)


if __name__ == '__main__':
    unittest.main()
