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

"""Quantum Generator."""

from typing import Optional, List, Union, Dict, Any
from copy import deepcopy
import numpy as np

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.aqua import aqua_globals
from qiskit.aqua.components.optimizers import ADAM
from qiskit.aqua.components.uncertainty_models import \
    UniformDistribution, MultivariateUniformDistribution
from qiskit.aqua.components.uncertainty_models import UnivariateVariationalDistribution, \
    MultivariateVariationalDistribution
from qiskit.aqua import AquaError
from qiskit.aqua.components.neural_networks.generative_network import GenerativeNetwork
from qiskit.aqua.components.initial_states import Custom

# pylint: disable=invalid-name


class QuantumGenerator(GenerativeNetwork):
    """Quantum Generator.

    The quantum generator is a parametrized quantum circuit which can be trained with the
    :class:`~qiskit.aqua.algorithms.QGAN` algorithm
    to generate a quantum state which approximates the probability
    distribution of given training data. At the beginning of the training the parameters will
    be set randomly, thus, the output will is random. Throughout the training the quantum
    generator learns to represent the target distribution.
    Eventually, the trained generator can be used for state preparation e.g. in QAE.
    """

    def __init__(self,
                 bounds: np.ndarray,
                 num_qubits: List[int],
                 generator_circuit: Optional[Union[UnivariateVariationalDistribution,
                                                   MultivariateVariationalDistribution,
                                                   QuantumCircuit]] = None,
                 init_params: Optional[Union[List[float], np.ndarray]] = None,
                 snapshot_dir: Optional[str] = None) -> None:
        """
        Args:
            bounds: k min/max data values [[min_1,max_1],...,[min_k,max_k]],
                given input data dim k
            num_qubits: k numbers of qubits to determine representation resolution,
                i.e. n qubits enable the representation of 2**n values [n_1,..., n_k]
            generator_circuit: a UnivariateVariationalDistribution for univariate data,
                a MultivariateVariationalDistribution for multivariate data,
                or a QuantumCircuit implementing the generator.
            init_params: 1D numpy array or list, Initialization for
                the generator's parameters.
            snapshot_dir: str or None, if not None save the optimizer's parameter after every
                update step to the given directory

        Raises:
            AquaError: Set multivariate variational distribution to represent multivariate data
        """
        super().__init__()
        self._bounds = bounds
        self._num_qubits = num_qubits
        self.generator_circuit = generator_circuit
        if self.generator_circuit is None:
            entangler_map = []
            if np.sum(num_qubits) > 2:
                for i in range(int(np.sum(num_qubits))):
                    entangler_map.append([i, int(np.mod(i + 1, np.sum(num_qubits)))])
            else:
                if np.sum(num_qubits) > 1:
                    entangler_map.append([0, 1])

            if len(num_qubits) > 1:
                num_qubits = list(map(int, num_qubits))
                low = bounds[:, 0].tolist()
                high = bounds[:, 1].tolist()
                init_dist = MultivariateUniformDistribution(num_qubits, low=low, high=high)
                q = QuantumRegister(sum(num_qubits))
                qc = QuantumCircuit(q)
                init_dist.build(qc, q)
                init_distribution = Custom(num_qubits=sum(num_qubits), circuit=qc)
                # Set variational form
                var_form = TwoLocal(sum(num_qubits), 'ry', 'cz', reps=1,
                                    initial_state=init_distribution,
                                    entanglement=entangler_map)
                if init_params is None:
                    init_params = aqua_globals.random.random(var_form.num_parameters) * 2 * 1e-2
                # Set generator circuit
                self.generator_circuit = MultivariateVariationalDistribution(num_qubits, var_form,
                                                                             init_params,
                                                                             low=low, high=high)
            else:
                init_dist = UniformDistribution(sum(num_qubits), low=bounds[0], high=bounds[1])
                q = QuantumRegister(sum(num_qubits), name='q')
                qc = QuantumCircuit(q)
                init_dist.build(qc, q)
                init_distribution = Custom(num_qubits=sum(num_qubits), circuit=qc)
                var_form = TwoLocal(sum(num_qubits), 'ry', 'cz', reps=1,
                                    initial_state=init_distribution,
                                    entanglement=entangler_map)
                if init_params is None:
                    init_params = aqua_globals.random.random(var_form.num_parameters) * 2 * 1e-2
                # Set generator circuit
                self.generator_circuit = UnivariateVariationalDistribution(
                    int(np.sum(num_qubits)), var_form, init_params, low=bounds[0], high=bounds[1])

        if len(num_qubits) > 1:
            if isinstance(self.generator_circuit, MultivariateVariationalDistribution):
                pass
            else:
                raise AquaError('Set multivariate variational distribution '
                                'to represent multivariate data')
        else:
            if isinstance(self.generator_circuit, UnivariateVariationalDistribution):
                pass
            else:
                raise AquaError('Set univariate variational distribution '
                                'to represent univariate data')
        # Set optimizer for updating the generator network
        self._optimizer = ADAM(maxiter=1, tol=1e-6, lr=1e-3, beta_1=0.7,
                               beta_2=0.99, noise_factor=1e-6,
                               eps=1e-6, amsgrad=True, snapshot_dir=snapshot_dir)

        if np.ndim(self._bounds) == 1:
            bounds = np.reshape(self._bounds, (1, len(self._bounds)))
        else:
            bounds = self._bounds
        for j, prec in enumerate(self._num_qubits):
            # prepare data grid for dim j
            grid = np.linspace(bounds[j, 0], bounds[j, 1], (2 ** prec))
            if j == 0:
                if len(self._num_qubits) > 1:
                    self._data_grid = [grid]
                else:
                    self._data_grid = grid
                self._grid_elements = grid
            elif j == 1:
                self._data_grid.append(grid)
                temp = []
                for g_e in self._grid_elements:
                    for g in grid:
                        temp0 = [g_e]
                        temp0.append(g)
                        temp.append(temp0)
                self._grid_elements = temp
            else:
                self._data_grid.append(grid)
                temp = []
                for g_e in self._grid_elements:
                    for g in grid:
                        temp0 = deepcopy(g_e)
                        temp0.append(g)
                        temp.append(temp0)
                self._grid_elements = deepcopy(temp)
        self._data_grid = np.array(self._data_grid)

        self._shots = None
        self._discriminator = None
        self._ret = {}  # type: Dict[str, Any]

    def set_seed(self, seed):
        """
        Set seed.

        Args:
            seed (int): seed
        """
        aqua_globals.random_seed = seed

    def set_discriminator(self, discriminator):
        """
        Set discriminator network.

        Args:
            discriminator (Discriminator): Discriminator used to compute the loss function.
        """
        self._discriminator = discriminator

    def construct_circuit(self, params=None):
        """
        Construct generator circuit.

        Args:
            params (numpy.ndarray): parameters which should be used to run the generator,
                    if None use self._params

        Returns:
            Instruction: construct the quantum circuit and return as gate
        """

        q = QuantumRegister(sum(self._num_qubits), name='q')
        qc = QuantumCircuit(q)
        if params is None:
            self.generator_circuit.build(qc=qc, q=q)
        else:
            generator_circuit_copy = deepcopy(self.generator_circuit)
            generator_circuit_copy.params = params
            generator_circuit_copy.build(qc=qc, q=q)

        # return qc.copy(name='qc')
        return qc.to_instruction()

    def get_output(self, quantum_instance, qc_state_in=None, params=None, shots=None):
        """
        Get classical data samples from the generator.
        Running the quantum generator circuit results in a quantum state.
        To train this generator with a classical discriminator, we need to sample classical outputs
        by measuring the quantum state and mapping them to feature space defined by the training
        data.

        Args:
            quantum_instance (QuantumInstance): Quantum Instance, used to run the generator
                circuit.
            qc_state_in (QuantumCircuit): deprecated
            params (numpy.ndarray): array or None, parameters which should
                be used to run the generator, if None use self._params
            shots (int): if not None use a number of shots that is different from the
                number set in quantum_instance

        Returns:
            list: generated samples, array: sample occurrence in percentage
        """
        instance_shots = quantum_instance.run_config.shots
        q = QuantumRegister(sum(self._num_qubits), name='q')
        qc = QuantumCircuit(q)
        qc.append(self.construct_circuit(params), q)
        if quantum_instance.is_statevector:
            pass
        else:
            c = ClassicalRegister(sum(self._num_qubits), name='c')
            qc.add_register(c)
            qc.measure(q, c)

        if shots is not None:
            quantum_instance.set_config(shots=shots)

        result = quantum_instance.execute(qc)

        generated_samples = []
        if quantum_instance.is_statevector:
            result = result.get_statevector(qc)
            values = np.multiply(result, np.conj(result))
            values = list(values.real)
            keys = []
            for j in range(len(values)):
                keys.append(np.binary_repr(j, int(sum(self._num_qubits))))
        else:
            result = result.get_counts(qc)
            keys = list(result)
            values = list(result.values())
            values = [float(v) / np.sum(values) for v in values]
        generated_samples_weights = values
        for i, _ in enumerate(keys):
            index = 0
            temp = []
            for k, p in enumerate(self._num_qubits):
                bin_rep = 0
                j = 0
                while j < p:
                    bin_rep += int(keys[i][index]) * 2 ** (int(p) - j - 1)
                    j += 1
                    index += 1
                if len(self._num_qubits) > 1:
                    temp.append(self._data_grid[k][int(bin_rep)])
                else:
                    temp.append(self._data_grid[int(bin_rep)])
            generated_samples.append(temp)

        self.generator_circuit._probabilities = generated_samples_weights
        if shots is not None:
            # Restore the initial quantum_instance configuration
            quantum_instance.set_config(shots=instance_shots)
        return generated_samples, generated_samples_weights

    def loss(self, x, weights):  # pylint: disable=arguments-differ
        """
        Loss function for training the generator's parameters.

        Args:
            x (numpy.ndarray): sample label (equivalent to discriminator output)
            weights (numpy.ndarray): probability for measuring the sample

        Returns:
            float: loss function
        """
        try:
            # pylint: disable=no-member
            loss = (-1) * np.dot(np.log(x).transpose(), weights)
        except Exception:  # pylint: disable=broad-except
            loss = (-1) * np.dot(np.log(x), weights)
        return loss.flatten()

    def _get_objective_function(self, quantum_instance, discriminator):
        """
        Get objective function

        Args:
            quantum_instance (QuantumInstance): used to run the quantum circuit.
            discriminator (torch.nn.Module): discriminator network to compute the sample labels.

        Returns:
            objective_function: objective function for quantum generator optimization
        """

        def objective_function(params):
            """
            Objective function

            Args:
                params (numpy.ndarray): generator parameters

            Returns:
                self.loss: loss function
            """
            generated_data, generated_prob = self.get_output(quantum_instance, params=params,
                                                             shots=self._shots)
            prediction_generated = discriminator.get_label(generated_data, detach=True)
            return self.loss(prediction_generated, generated_prob)

        return objective_function

    def train(self, quantum_instance=None, shots=None):
        """
        Perform one training step w.r.t to the generator's parameters

        Args:
            quantum_instance (QuantumInstance): used to run the generator circuit.
            shots (int): Number of shots for hardware or qasm execution.

        Returns:
            dict: generator loss(float) and updated parameters (array).
        """

        self._shots = shots
        # Force single optimization iteration
        self._optimizer._maxiter = 1
        self._optimizer._t = 0
        objective = self._get_objective_function(quantum_instance, self._discriminator)
        self.generator_circuit.params, loss, _ = self._optimizer.optimize(
            num_vars=len(self.generator_circuit.params),
            objective_function=objective,
            initial_point=self.generator_circuit.params
            )

        self._ret['loss'] = loss
        self._ret['params'] = self.generator_circuit.params

        return self._ret
