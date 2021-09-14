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

"""Quantum Generator."""

from typing import Optional, List, Union, Dict, Any, Callable, cast, Tuple
from copy import deepcopy
import warnings

import numpy as np

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms.optimizers import ADAM, Optimizer
from qiskit.opflow.gradients import Gradient
from qiskit.opflow import CircuitStateFn
from ....exceptions import QiskitMachineLearningError
from .generative_network import GenerativeNetwork
from .discriminative_network import DiscriminativeNetwork

# pylint: disable=invalid-name


class QuantumGenerator(GenerativeNetwork):
    """Quantum Generator.

    The quantum generator is a parametrized quantum circuit which can be trained with the
    :class:`~qiskit_machine_learning.algorithms.QGAN` algorithm
    to generate a quantum state which approximates the probability
    distribution of given training data. At the beginning of the training the parameters will
    be set randomly, thus, the output will is random. Throughout the training the quantum
    generator learns to represent the target distribution.
    Eventually, the trained generator can be used for state preparation e.g. in QAE.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        num_qubits: Union[List[int], np.ndarray],
        generator_circuit: Optional[QuantumCircuit] = None,
        init_params: Optional[Union[List[float], np.ndarray]] = None,
        optimizer: Optional[Optimizer] = None,
        gradient_function: Optional[Union[Callable, Gradient]] = None,
        snapshot_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            bounds: k min/max data values [[min_1,max_1],...,[min_k,max_k]],
                given input data dim k
            num_qubits: k numbers of qubits to determine representation resolution,
                i.e. n qubits enable the representation of 2**n values [n_1,..., n_k]
            generator_circuit: a QuantumCircuit implementing the generator.
            init_params: 1D numpy array or list, Initialization for
                the generator's parameters.
            optimizer: optimizer to be used for the training of the generator
            gradient_function: A Gradient object, or a function returning partial
                derivatives of the loss function w.r.t. the generator variational
                params.
            snapshot_dir: str or None, if not None save the optimizer's parameter after every
                update step to the given directory

        Raises:
            QiskitMachineLearningError: Set multivariate variational distribution
                                        to represent multivariate data
        """
        super().__init__()
        self._bounds = bounds
        self._num_qubits = num_qubits
        self.generator_circuit = generator_circuit
        if generator_circuit is None:
            circuit = QuantumCircuit(sum(num_qubits))
            circuit.h(circuit.qubits)
            ansatz = TwoLocal(sum(num_qubits), "ry", "cz", reps=1, entanglement="circular")
            circuit.compose(ansatz, inplace=True)

            # Set generator circuit
            self.generator_circuit = circuit

        self._free_parameters = sorted(self.generator_circuit.parameters, key=lambda p: p.name)

        if init_params is None:
            init_params = (
                algorithm_globals.random.random(self.generator_circuit.num_parameters) * 2e-2
            )

        self._bound_parameters = init_params

        # Set optimizer for updating the generator network
        self._snapshot_dir = snapshot_dir
        self.optimizer = optimizer

        self._gradient_function = gradient_function

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
                    self._data_grid = grid  # type: ignore
                self._grid_elements = grid
            elif j == 1:
                self._data_grid.append(grid)
                temp = []
                for g_e in self._grid_elements:
                    for g in grid:
                        temp0 = [g_e]
                        temp0.append(g)
                        temp.append(temp0)
                self._grid_elements = temp  # type: ignore
            else:
                self._data_grid.append(grid)
                temp = []
                for g_e in self._grid_elements:
                    for g in grid:
                        temp0 = deepcopy(g_e)
                        temp0.append(g)
                        temp.append(temp0)
                self._grid_elements = deepcopy(temp)  # type: ignore
        self._data_grid = np.array(self._data_grid, dtype=object)  # type: ignore

        self._seed = 7
        self._shots = None
        self._discriminator: Optional[DiscriminativeNetwork] = None
        self._ret: Dict[str, Any] = {}

    @property
    def parameter_values(self) -> Union[List, np.ndarray]:
        """
        Get parameter values from the quantum generator

        Returns:
            Current parameter values
        """
        return self._bound_parameters

    @parameter_values.setter
    def parameter_values(self, p_values: Union[List, np.ndarray]) -> None:
        """
        Set parameter values for the quantum generator

        Args:
            p_values: Parameter values
        """
        self._bound_parameters = p_values

    @property
    def seed(self) -> int:
        """
        Get seed.
        """
        return self._seed

    @seed.setter
    def seed(self, seed: int) -> None:
        """
        Set seed.

        Args:
            seed (int): seed to use.
        """
        self._seed = seed
        algorithm_globals.random_seed = seed

    @property
    def discriminator(self) -> DiscriminativeNetwork:
        """
        Get discriminator.
        """
        return self._discriminator

    @discriminator.setter
    def discriminator(self, discriminator: DiscriminativeNetwork) -> None:
        """
        Set discriminator.

        Args:
            discriminator (DiscriminativeNetwork): Discriminator used to
                compute the loss function.
        """
        self._discriminator = discriminator

    @property
    def optimizer(self) -> Optimizer:
        """
        Get optimizer.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optional[Optimizer] = None) -> None:
        """
        Set optimizer.

        Args:
            optimizer (Optimizer): optimizer to use with the generator.

        Raises:
            QiskitMachineLearningError: invalid input.
        """
        if optimizer:
            if isinstance(optimizer, Optimizer):
                self._optimizer = optimizer
            else:
                raise QiskitMachineLearningError(
                    "Please provide an Optimizer object to use as the generator optimizer."
                )
        else:
            self._optimizer = ADAM(
                maxiter=1,
                tol=1e-6,
                lr=1e-3,
                beta_1=0.7,
                beta_2=0.99,
                noise_factor=1e-6,
                eps=1e-6,
                amsgrad=True,
                snapshot_dir=self._snapshot_dir,
            )

    def construct_circuit(self, params=None):
        """
        Construct generator circuit.

        Args:
            params (list | dict): parameters which should be used to run the generator.

        Returns:
            Instruction: construct the quantum circuit and return as gate
        """
        if params is None:
            return self.generator_circuit

        if isinstance(params, (list, np.ndarray)):
            params = dict(zip(self._free_parameters, params))

        return self.generator_circuit.assign_parameters(params)
        #     self.generator_circuit.build(qc=qc, q=q)
        # else:
        #     generator_circuit_copy = deepcopy(self.generator_circuit)
        #     generator_circuit_copy.params = params
        #     generator_circuit_copy.build(qc=qc, q=q)

        # # return qc.copy(name='qc')
        # return qc.to_instruction()

    def get_output(
        self,
        quantum_instance: QuantumInstance,
        params: Optional[np.ndarray] = None,
        shots: Optional[int] = None,
    ) -> Tuple[List, List]:
        """
        Get classical data samples from the generator.
        Running the quantum generator circuit results in a quantum state.
        To train this generator with a classical discriminator, we need to sample classical outputs
        by measuring the quantum state and mapping them to feature space defined by the training
        data.

        Args:
            quantum_instance: Quantum Instance, used to run the generator
                circuit.
            params: array or None, parameters which should
                be used to run the generator, if None use self._params
            shots: if not None use a number of shots that is different from the
                number set in quantum_instance

        Returns:
            generated samples, array: sample occurrence in percentage
        """
        instance_shots = quantum_instance.run_config.shots
        q = QuantumRegister(sum(self._num_qubits), name="q")
        qc = QuantumCircuit(q)
        if params is None:
            params = cast(np.ndarray, self._bound_parameters)
        qc.append(self.construct_circuit(params), q)
        if quantum_instance.is_statevector:
            pass
        else:
            c = ClassicalRegister(sum(self._num_qubits), name="c")
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

        # self.generator_circuit._probabilities = generated_samples_weights
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
            generated_data, generated_prob = self.get_output(
                quantum_instance, params=params, shots=self._shots
            )
            prediction_generated = discriminator.get_label(generated_data, detach=True)
            return self.loss(prediction_generated, generated_prob)

        return objective_function

    def _convert_to_gradient_function(self, gradient_object, quantum_instance, discriminator):
        """
        Convert to gradient function

        Args:
            gradient_object (Gradient): the gradient object to be used to
                compute analytical gradients.
            quantum_instance (QuantumInstance): used to run the quantum circuit.
            discriminator (torch.nn.Module): discriminator network to compute the sample labels.

        Returns:
            gradient_function: gradient function that takes the current
                parameter values and returns partial derivatives of the loss
                function w.r.t. the variational parameters.
        """

        def gradient_function(current_point):
            """
            Gradient function

            Args:
                current_point (np.ndarray): Current values for the variational parameters.

            Returns:
                np.ndarray: array of partial derivatives of the loss
                    function w.r.t. the variational parameters.
            """
            free_params = self._free_parameters
            generated_data, _ = self.get_output(
                quantum_instance, params=current_point, shots=self._shots
            )
            prediction_generated = discriminator.get_label(generated_data, detach=True)
            op = ~CircuitStateFn(primitive=self.generator_circuit)
            grad_object = gradient_object.convert(operator=op, params=free_params)
            value_dict = {free_params[i]: current_point[i] for i in range(len(free_params))}
            analytical_gradients = np.array(grad_object.assign_parameters(value_dict).eval())
            loss_gradients = self.loss(
                prediction_generated, np.transpose(analytical_gradients)
            ).real
            return loss_gradients

        return gradient_function

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

        # TODO Improve access to maxiter, say via options getter, to avoid private member access
        # and since not all optimizers have that exact naming figure something better as well to
        # allow the checking below to not have to warn if it has something else and max iterations
        # is truly 1 anyway.
        try:
            if self._optimizer._maxiter != 1:
                warnings.warn(
                    "Please set the the optimizer maxiter argument to 1 "
                    "to ensure that the generator "
                    "and discriminator are updated in an alternating fashion."
                )
        except AttributeError:
            maxiter = self._optimizer._options.get("maxiter")
            if maxiter is not None and maxiter != 1:
                warnings.warn(
                    "Please set the the optimizer maxiter argument to 1 "
                    "to ensure that the generator "
                    "and discriminator are updated in an alternating fashion."
                )
            elif maxiter is None:
                warnings.warn(
                    "Please ensure the optimizer max iterations are set to 1 "
                    "to ensure that the generator "
                    "and discriminator are updated in an alternating fashion."
                )

        if isinstance(self._gradient_function, Gradient):
            self._gradient_function = self._convert_to_gradient_function(
                self._gradient_function, quantum_instance, self._discriminator
            )

        objective = self._get_objective_function(quantum_instance, self._discriminator)
        opt_result = self._optimizer.minimize(
            fun=objective,
            x0=self._bound_parameters,
            jac=self._gradient_function,
        )
        self._bound_parameters = opt_result.x
        self._ret["loss"] = opt_result.fun
        self._ret["params"] = self._bound_parameters

        return self._ret
