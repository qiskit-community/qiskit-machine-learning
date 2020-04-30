# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test QGAN """

from test.aqua import QiskitAquaTestCase

import warnings
import unittest
from ddt import ddt, data
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.aqua.components.uncertainty_models import (UniformDistribution,
                                                       UnivariateVariationalDistribution)
from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua.algorithms import QGAN
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.components.initial_states import Custom
from qiskit.aqua.components.neural_networks import NumPyDiscriminator, PyTorchDiscriminator
from qiskit import BasicAer


@ddt
class TestQGAN(QiskitAquaTestCase):
    """ Test QGAN """

    def setUp(self):
        super().setUp()

        self.seed = 7
        aqua_globals.random_seed = self.seed
        # Number training data samples
        n_v = 5000
        # Load data samples from log-normal distribution with mean=1 and standard deviation=1
        m_u = 1
        sigma = 1
        self._real_data = aqua_globals.random.lognormal(mean=m_u, sigma=sigma, size=n_v)
        # Set the data resolution
        # Set upper and lower data values as list of k
        # min/max data values [[min_0,max_0],...,[min_k-1,max_k-1]]
        self._bounds = [0., 3.]
        # Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
        num_qubits = [2]
        # Batch size
        batch_size = 100
        # Set number of training epochs
        # num_epochs = 10
        num_epochs = 5

        # Initialize qGAN
        self.qgan = QGAN(self._real_data,
                         self._bounds,
                         num_qubits,
                         batch_size,
                         num_epochs,
                         snapshot_dir=None)
        self.qgan.seed = 7
        # Set quantum instance to run the quantum generator
        self.qi_statevector = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                              seed_simulator=2,
                                              seed_transpiler=2)
        self.qi_qasm = QuantumInstance(backend=BasicAer.get_backend('qasm_simulator'),
                                       shots=1000,
                                       seed_simulator=2,
                                       seed_transpiler=2)
        # Set entangler map
        entangler_map = [[0, 1]]

        # Set an initial state for the generator circuit
        init_dist = UniformDistribution(sum(num_qubits), low=self._bounds[0], high=self._bounds[1])
        q = QuantumRegister(sum(num_qubits), name='q')
        qc = QuantumCircuit(q)
        init_dist.build(qc, q)
        init_distribution = Custom(num_qubits=sum(num_qubits), circuit=qc)

        # Set generator's initial parameters
        init_params = aqua_globals.random.rand(2 * sum(num_qubits)) * 2 * 1e-2

        # Set variational form
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        var_form = RY(sum(num_qubits),
                      depth=1,
                      initial_state=init_distribution,
                      entangler_map=entangler_map,
                      entanglement_gate='cx')
        dist_var_form = UnivariateVariationalDistribution(sum(num_qubits), var_form, init_params,
                                                          low=self._bounds[0],
                                                          high=self._bounds[1])
        warnings.filterwarnings('always', category=DeprecationWarning)

        library = RealAmplitudes(sum(num_qubits), reps=1, initial_state=init_distribution,
                                 entanglement=entangler_map)
        dist_library = UnivariateVariationalDistribution(sum(num_qubits), library, init_params,
                                                         low=self._bounds[0],
                                                         high=self._bounds[1])
        circuit = QuantumCircuit(sum(num_qubits)).compose(library)
        dist_circuit = UnivariateVariationalDistribution(sum(num_qubits), circuit, init_params,
                                                         low=self._bounds[0],
                                                         high=self._bounds[1])

        self.generator_circuits = {'wrapped': dist_var_form,
                                   'circuit': dist_circuit,
                                   'library': dist_library}

    @data('wrapped', 'circuit', 'library')
    def test_sample_generation(self, mode):
        """ sample generation test """
        if mode == 'wrapped':
            # ignore deprecation warnings from the deprecation of VariationalForm as input for
            # the univariate variational distribution
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            self.qgan.set_generator(generator_circuit=self.generator_circuits[mode])
            warnings.filterwarnings('always', category=DeprecationWarning)
        else:
            self.qgan.set_generator(generator_circuit=self.generator_circuits[mode])

        _, weights_statevector = \
            self.qgan._generator.get_output(self.qi_statevector, shots=100)
        samples_qasm, weights_qasm = self.qgan._generator.get_output(self.qi_qasm, shots=100)
        samples_qasm, weights_qasm = zip(*sorted(zip(samples_qasm, weights_qasm)))
        for i, weight_q in enumerate(weights_qasm):
            self.assertAlmostEqual(weight_q, weights_statevector[i], delta=0.1)

    @data('wrapped', 'circuit', 'library')
    def test_qgan_training(self, mode):
        """ qgan training test """
        if mode == 'wrapped':
            # ignore deprecation warnings from the deprecation of VariationalForm as input for
            # the univariate variational distribution
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            self.qgan.set_generator(generator_circuit=self.generator_circuits[mode])
            warnings.filterwarnings('always', category=DeprecationWarning)
        else:
            self.qgan.set_generator(generator_circuit=self.generator_circuits[mode])

        trained_statevector = self.qgan.run(self.qi_statevector)
        trained_qasm = self.qgan.run(self.qi_qasm)
        self.assertAlmostEqual(trained_qasm['rel_entr'], trained_statevector['rel_entr'], delta=0.1)

    def test_qgan_training_run_algo_torch(self):
        """ qgan training run algo torch test """
        try:
            # Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
            num_qubits = [2]
            # Batch size
            batch_size = 100
            # Set number of training epochs
            num_epochs = 5
            _qgan = QGAN(self._real_data,
                         self._bounds,
                         num_qubits,
                         batch_size,
                         num_epochs,
                         discriminator=PyTorchDiscriminator(n_features=len(num_qubits)),
                         snapshot_dir=None)
            _qgan.seed = self.seed
            _qgan.set_generator()
            trained_statevector = _qgan.run(QuantumInstance(
                BasicAer.get_backend('statevector_simulator'),
                seed_simulator=aqua_globals.random_seed,
                seed_transpiler=aqua_globals.random_seed))
            trained_qasm = _qgan.run(QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                                     seed_simulator=aqua_globals.random_seed,
                                                     seed_transpiler=aqua_globals.random_seed))
            self.assertAlmostEqual(trained_qasm['rel_entr'],
                                   trained_statevector['rel_entr'], delta=0.1)
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest(str(ex))

    def test_qgan_training_run_algo_numpy(self):
        """ qgan training run algo numpy test """
        # Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
        num_qubits = [2]
        # Batch size
        batch_size = 100
        # Set number of training epochs
        num_epochs = 5
        _qgan = QGAN(self._real_data,
                     self._bounds,
                     num_qubits,
                     batch_size,
                     num_epochs,
                     discriminator=NumPyDiscriminator(n_features=len(num_qubits)),
                     snapshot_dir=None)
        _qgan.seed = self.seed
        _qgan.set_generator()
        trained_statevector = _qgan.run(
            QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                            seed_simulator=aqua_globals.random_seed,
                            seed_transpiler=aqua_globals.random_seed))
        trained_qasm = _qgan.run(QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                                 seed_simulator=aqua_globals.random_seed,
                                                 seed_transpiler=aqua_globals.random_seed))
        self.assertAlmostEqual(trained_qasm['rel_entr'], trained_statevector['rel_entr'], delta=0.1)


if __name__ == '__main__':
    unittest.main()
