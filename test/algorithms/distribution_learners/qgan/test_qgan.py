# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the QGAN algorithm."""

import unittest
import warnings
import tempfile
from test import QiskitMachineLearningTestCase, requires_extra_library

from ddt import ddt, data

from qiskit import BasicAer
from qiskit.circuit.library import UniformDistribution, RealAmplitudes
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms.optimizers import CG, COBYLA
from qiskit.opflow.gradients import Gradient
from qiskit_machine_learning.algorithms import (
    NumPyDiscriminator,
    PyTorchDiscriminator,
    QGAN,
)


@ddt
class TestQGAN(QiskitMachineLearningTestCase):
    """Test the QGAN algorithm."""

    def setUp(self):
        super().setUp()

        self.seed = 7
        algorithm_globals.random_seed = self.seed
        # Number training data samples
        n_v = 5000
        # Load data samples from log-normal distribution with mean=1 and standard deviation=1
        m_u = 1
        sigma = 1
        self._real_data = algorithm_globals.random.lognormal(mean=m_u, sigma=sigma, size=n_v)
        # Set upper and lower data values as list of k
        # min/max data values [[min_0,max_0],...,[min_k-1,max_k-1]]
        self._bounds = [0.0, 3.0]
        # Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
        num_qubits = [2]
        # Batch size
        batch_size = 100
        # Set number of training epochs
        # num_epochs = 10
        num_epochs = 5

        # Initialize qGAN
        self.qgan = QGAN(
            self._real_data,
            self._bounds,
            num_qubits,
            batch_size,
            num_epochs,
            snapshot_dir=None,
        )
        self.qgan.seed = 7
        # Set quantum instance to run the quantum generator
        self.qi_statevector = QuantumInstance(
            backend=BasicAer.get_backend("statevector_simulator"),
            seed_simulator=2,
            seed_transpiler=2,
        )
        self.qi_qasm = QuantumInstance(
            backend=BasicAer.get_backend("qasm_simulator"),
            shots=1000,
            seed_simulator=2,
            seed_transpiler=2,
        )
        # Set entangler map
        entangler_map = [[0, 1]]

        qc = UniformDistribution(sum(num_qubits))

        ansatz = RealAmplitudes(sum(num_qubits), reps=1, entanglement=entangler_map)
        qc.compose(ansatz, inplace=True)
        self.generator_circuit = qc

    def test_sample_generation(self):
        """Test sample generation."""
        self.qgan.set_generator(generator_circuit=self.generator_circuit)

        _, weights_statevector = self.qgan._generator.get_output(self.qi_statevector, shots=100)
        samples_qasm, weights_qasm = self.qgan._generator.get_output(self.qi_qasm, shots=100)
        samples_qasm, weights_qasm = zip(*sorted(zip(samples_qasm, weights_qasm)))
        for i, weight_q in enumerate(weights_qasm):
            self.assertAlmostEqual(weight_q, weights_statevector[i], delta=0.1)

    def test_qgan_training_cg(self):
        """Test QGAN training."""
        optimizer = CG(maxiter=1)
        self.qgan.set_generator(
            generator_circuit=self.generator_circuit, generator_optimizer=optimizer
        )
        trained_statevector = self.qgan.run(self.qi_statevector)
        trained_qasm = self.qgan.run(self.qi_qasm)
        self.assertAlmostEqual(trained_qasm["rel_entr"], trained_statevector["rel_entr"], delta=0.1)

    def test_qgan_training_cobyla(self):
        """Test QGAN training."""
        optimizer = COBYLA(maxiter=1)
        self.qgan.set_generator(
            generator_circuit=self.generator_circuit, generator_optimizer=optimizer
        )
        trained_statevector = self.qgan.run(self.qi_statevector)
        trained_qasm = self.qgan.run(self.qi_qasm)
        self.assertAlmostEqual(trained_qasm["rel_entr"], trained_statevector["rel_entr"], delta=0.1)

    def test_qgan_training(self):
        """Test QGAN training."""
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.qgan.set_generator(generator_circuit=self.generator_circuit)
        warnings.filterwarnings("always", category=DeprecationWarning)

        trained_statevector = self.qgan.run(self.qi_statevector)
        trained_qasm = self.qgan.run(self.qi_qasm)
        self.assertAlmostEqual(trained_qasm["rel_entr"], trained_statevector["rel_entr"], delta=0.1)

    @requires_extra_library
    def test_qgan_training_run_algo_torch(self):
        """Test QGAN training using a PyTorch discriminator."""
        # Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
        num_qubits = [2]
        # Batch size
        batch_size = 100
        # Set number of training epochs
        num_epochs = 5
        _qgan = QGAN(
            self._real_data,
            self._bounds,
            num_qubits,
            batch_size,
            num_epochs,
            discriminator=PyTorchDiscriminator(n_features=len(num_qubits)),
            snapshot_dir=None,
        )
        _qgan.seed = self.seed
        _qgan.set_generator()
        trained_statevector = _qgan.run(
            QuantumInstance(
                BasicAer.get_backend("statevector_simulator"),
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            )
        )
        trained_qasm = _qgan.run(
            QuantumInstance(
                BasicAer.get_backend("qasm_simulator"),
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            )
        )
        self.assertAlmostEqual(trained_qasm["rel_entr"], trained_statevector["rel_entr"], delta=0.1)

    @requires_extra_library
    def test_qgan_training_run_algo_torch_multivariate(self):
        """Test QGAN training using a PyTorch discriminator, for multivariate distributions."""
        # Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
        num_qubits = [1, 2]
        # Batch size
        batch_size = 100
        # Set number of training epochs
        num_epochs = 5

        # Reshape data in a multi-variate fashion
        # (two independent identically distributed variables,
        # each represented by half of the generated samples)
        real_data = self._real_data.reshape((-1, 2))
        bounds = [self._bounds, self._bounds]

        _qgan = QGAN(
            real_data,
            bounds,
            num_qubits,
            batch_size,
            num_epochs,
            discriminator=PyTorchDiscriminator(n_features=len(num_qubits)),
            snapshot_dir=None,
        )
        _qgan.seed = self.seed
        _qgan.set_generator()
        trained_statevector = _qgan.run(
            QuantumInstance(
                BasicAer.get_backend("statevector_simulator"),
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            )
        )
        trained_qasm = _qgan.run(
            QuantumInstance(
                BasicAer.get_backend("qasm_simulator"),
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            )
        )
        self.assertAlmostEqual(trained_qasm["rel_entr"], trained_statevector["rel_entr"], delta=0.1)

    def test_qgan_training_run_algo_numpy(self):
        """Test QGAN training using a NumPy discriminator."""
        # Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
        num_qubits = [2]
        # Batch size
        batch_size = 100
        # Set number of training epochs
        num_epochs = 5
        _qgan = QGAN(
            self._real_data,
            self._bounds,
            num_qubits,
            batch_size,
            num_epochs,
            discriminator=NumPyDiscriminator(n_features=len(num_qubits)),
            snapshot_dir=None,
        )
        _qgan.seed = self.seed
        _qgan.set_generator()
        trained_statevector = _qgan.run(
            QuantumInstance(
                BasicAer.get_backend("statevector_simulator"),
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            )
        )
        trained_qasm = _qgan.run(
            QuantumInstance(
                BasicAer.get_backend("qasm_simulator"),
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            )
        )
        self.assertAlmostEqual(trained_qasm["rel_entr"], trained_statevector["rel_entr"], delta=0.1)

    def test_qgan_save_model(self):
        """Test the QGAN functionality to store the current model."""
        # Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
        num_qubits = [2]
        # Batch size
        batch_size = 100
        # Set number of training epochs
        num_epochs = 5
        with tempfile.TemporaryDirectory() as tmpdirname:
            _qgan = QGAN(
                self._real_data,
                self._bounds,
                num_qubits,
                batch_size,
                num_epochs,
                discriminator=NumPyDiscriminator(n_features=len(num_qubits)),
                snapshot_dir=tmpdirname,
            )
            _qgan.seed = self.seed
            _qgan.set_generator()
            trained_statevector = _qgan.run(
                QuantumInstance(
                    BasicAer.get_backend("statevector_simulator"),
                    seed_simulator=algorithm_globals.random_seed,
                    seed_transpiler=algorithm_globals.random_seed,
                )
            )
            trained_qasm = _qgan.run(
                QuantumInstance(
                    BasicAer.get_backend("qasm_simulator"),
                    seed_simulator=algorithm_globals.random_seed,
                    seed_transpiler=algorithm_globals.random_seed,
                )
            )
        self.assertAlmostEqual(trained_qasm["rel_entr"], trained_statevector["rel_entr"], delta=0.1)

    def test_qgan_training_run_algo_numpy_multivariate(self):
        """Test QGAN training using a NumPy discriminator, for multivariate distributions."""
        # Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
        num_qubits = [1, 2]
        # Batch size
        batch_size = 100
        # Set number of training epochs
        num_epochs = 5

        # Reshape data in a multi-variate fashion
        # (two independent identically distributed variables,
        # each represented by half of the generated samples)
        real_data = self._real_data.reshape((-1, 2))
        bounds = [self._bounds, self._bounds]

        _qgan = QGAN(
            real_data,
            bounds,
            num_qubits,
            batch_size,
            num_epochs,
            discriminator=NumPyDiscriminator(n_features=len(num_qubits)),
            snapshot_dir=None,
        )
        _qgan.seed = self.seed
        _qgan.set_generator()
        trained_statevector = _qgan.run(
            QuantumInstance(
                BasicAer.get_backend("statevector_simulator"),
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            )
        )
        trained_qasm = _qgan.run(
            QuantumInstance(
                BasicAer.get_backend("qasm_simulator"),
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            )
        )
        self.assertAlmostEqual(trained_qasm["rel_entr"], trained_statevector["rel_entr"], delta=0.1)

    @data("qasm", "sv")
    def test_qgan_training_analytic_gradients(self, backend: str):
        """
        Test QGAN with analytic gradients
        Args:
            backend: backend to run the training
        """
        if backend == "qasm":
            q_inst = self.qi_qasm
        else:
            q_inst = self.qi_statevector
        self.qgan.set_generator(self.generator_circuit)
        numeric_results = self.qgan.run(q_inst)
        self.qgan.set_generator(self.generator_circuit, generator_gradient=Gradient("param_shift"))
        analytic_results = self.qgan.run(q_inst)
        self.assertAlmostEqual(numeric_results["rel_entr"], analytic_results["rel_entr"], delta=0.1)


if __name__ == "__main__":
    unittest.main()
