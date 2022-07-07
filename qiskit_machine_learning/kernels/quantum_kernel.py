# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Kernel Algorithm"""
from __future__ import annotations

import copy
import numbers
from typing import Optional, Union, Sequence, Mapping, List, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.providers import Backend
from qiskit.result import Result
from qiskit.utils import QuantumInstance

from qiskit_machine_learning.deprecation import (
    deprecate_arguments,
    deprecate_method,
    deprecate_property,
)
from ..exceptions import QiskitMachineLearningError


class QuantumKernel:
    r"""Quantum Kernel.

    The general task of machine learning is to find and study patterns in data. For many
    algorithms, the datapoints are better understood in a higher dimensional feature space,
    through the use of a kernel function:

    .. math::

        K(x, y) = \langle f(x), f(y)\rangle.

    Here K is the kernel function, x, y are n dimensional inputs. f is a map from n-dimension
    to m-dimension space. :math:`\langle x, y \rangle` denotes the dot product.
    Usually m is much larger than n.

    The quantum kernel algorithm calculates a kernel matrix, given datapoints x and y and feature
    map f, all of n dimension. This kernel matrix can then be used in classical machine learning
    algorithms such as support vector classification, spectral clustering or ridge regression.
    """

    @deprecate_arguments("0.5.0", {"user_parameters": "training_parameters"})
    def __init__(
        self,
        feature_map: Optional[QuantumCircuit] = None,
        enforce_psd: bool = True,
        batch_size: int = 900,
        quantum_instance: Optional[Union[QuantumInstance, Backend]] = None,
        training_parameters: Optional[Union[ParameterVector, Sequence[Parameter]]] = None,
    ) -> None:
        """
        Args:
            feature_map: Parameterized circuit to be used as the feature map. If None is given,
                the `ZZFeatureMap` is used with two qubits.
            enforce_psd: Project to closest positive semidefinite matrix if x = y.
                Only enforced when not using the state vector simulator. Default True.
            batch_size: Number of circuits to batch together for computation. Default 900.
            quantum_instance: Quantum Instance or Backend
            training_parameters: Iterable containing ``Parameter`` objects which correspond to
                 quantum gates on the feature map circuit which may be tuned. If users intend to
                 tune feature map parameters to find optimal values, this field should be set.
        """
        # Class fields
        self._feature_map = None
        self._unbound_feature_map = None
        self._training_parameters = None
        self._training_parameter_binds = None
        self._enforce_psd = enforce_psd
        self._batch_size = batch_size
        # convert to QuantumInstance if an instance of Backend is passed
        self.quantum_instance = quantum_instance

        # Setters
        self.feature_map = feature_map if feature_map is not None else ZZFeatureMap(2)
        if training_parameters is not None:
            self.training_parameters = training_parameters

    @property
    def feature_map(self) -> QuantumCircuit:
        """Return feature map"""
        return self._feature_map

    @feature_map.setter
    def feature_map(self, feature_map: QuantumCircuit) -> None:
        """
        Set feature map.

        The ``unbound_feature_map`` field will be automatically updated when this field is set,
        and ``training_parameters`` and ``training_parameter_binds`` fields will be reset to ``None``.
        """
        self._feature_map = feature_map
        self._unbound_feature_map = copy.deepcopy(self._feature_map)
        self._training_parameters = None
        self._training_parameter_binds = None

    @property
    def unbound_feature_map(self) -> QuantumCircuit:
        """Return unbound feature map"""
        return copy.deepcopy(self._unbound_feature_map)

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Return quantum instance"""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Union[Backend, QuantumInstance]) -> None:
        """Set quantum instance"""
        if isinstance(quantum_instance, Backend):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance

    @property
    def training_parameters(self) -> Optional[Union[ParameterVector, Sequence[Parameter]]]:
        """Return the vector of training parameters."""
        return copy.copy(self._training_parameters)

    @training_parameters.setter
    def training_parameters(
        self, training_params: Union[ParameterVector, Sequence[Parameter]]
    ) -> None:
        """Set the training parameters"""
        self._training_parameter_binds = {
            training_param: training_param for training_param in training_params
        }
        self._training_parameters = copy.deepcopy(training_params)

    def assign_training_parameters(
        self, values: Union[Mapping[Parameter, ParameterValueType], Sequence[ParameterValueType]]
    ) -> None:
        """
        Assign training parameters in the ``QuantumKernel`` feature map.

        Args:
            values (dict or iterable): Either a dictionary or iterable specifying the new
            parameter values. If a dict, it specifies the mapping from ``current_parameter`` to
            ``new_parameter``, where ``new_parameter`` can be a parameter expression or a
            numeric value. If an iterable, the elements are assigned to the existing parameters
            in the order of ``QuantumKernel.training_parameters``.

        Raises:
            ValueError: Incompatible number of training parameters and values

        """
        if self._training_parameters is None:
            raise ValueError(
                f"""
                The number of parameter values ({len(values)}) does not
                match the number of training parameters tracked by the QuantumKernel
                (None).
                """
            )

        # Get the input parameters. These should remain unaffected by assigning of training parameters.
        input_params = list(
            set(self._unbound_feature_map.parameters) - set(self._training_parameters)
        )

        # If iterable of values is passed, the length must match length of training_parameters field
        if isinstance(values, (Sequence, np.ndarray)):
            if len(values) != len(self._training_parameters):
                raise ValueError(
                    f"""
                The number of parameter values ({len(values)}) does not
                match the number of training parameters tracked by the QuantumKernel
                ({len(self._training_parameters)}).
                """
                )
            values = {p: values[i] for i, p in enumerate(self._training_parameters)}
        else:
            if not isinstance(values, dict):
                raise ValueError(
                    f"""
                    'values' must be of type Dict or Sequence.
                    Type {type(values)} is not supported.
                    """
                )

            # All input keys must exist in the circuit
            # This check actually catches some well defined assignments;
            # however; we throw an error to be consistent with the behavior
            # of QuantumCircuit's parameter binding.
            unknown_parameters = list(set(values.keys()) - set(self._training_parameters))
            if len(unknown_parameters) > 0:
                raise ValueError(
                    f"Cannot bind parameters ({unknown_parameters}) not tracked by the quantum kernel."
                )

        # Because QuantumKernel supports parameter rebinding, entries of the `values` dictionary must
        # be handled differently depending on whether they represent numerical assignments or parameter
        # reassignments. However, re-ordering the values dictionary inherently changes the expected
        # behavior of parameter binding, as entries in the values dict do not commute with one another
        # in general. To resolve this issue, we handle each entry of the values dict one at a time.
        for param, bind in values.items():
            if isinstance(bind, ParameterExpression):
                self._unbound_feature_map.assign_parameters({param: bind}, inplace=True)

                # Training params are all non-input params in the unbound feature map
                # This list comprehension ensures that self._training_parameters is ordered
                # in a way that is consistent with self.feature_map.parameters
                self._training_parameters = [
                    p for p in self._unbound_feature_map.parameters if (p not in input_params)
                ]

                # Remove param if it was overwritten
                if param not in self._training_parameters:
                    del self._training_parameter_binds[param]

                # Add new parameters
                for sub_param in bind.parameters:
                    if sub_param not in self._training_parameter_binds.keys():
                        self._training_parameter_binds[sub_param] = sub_param

                # If parameter is being set to expression of itself, training_parameter_binds
                # reflects a self-bind
                if param in bind.parameters:
                    self._training_parameter_binds[param] = param

            # If assignment is numerical, update the param_binds
            elif isinstance(bind, numbers.Number):
                self._training_parameter_binds[param] = bind

            else:
                raise ValueError(
                    f"""
                    Parameters can only be bound to numeric values,
                    Parameters, or ParameterExpressions. Type {type(bind)}
                    is not supported.
                    """
                )

        # Reorder dict according to self._training_parameters
        self._training_parameter_binds = {
            param: self._training_parameter_binds[param] for param in self._training_parameters
        }

        # Update feature map with numerical parameter assignments
        self._feature_map = self._unbound_feature_map.assign_parameters(
            self._training_parameter_binds
        )

    @property
    def training_parameter_binds(self) -> Optional[Mapping[Parameter, float]]:
        """Return a copy of the current training parameter mappings for the feature map circuit."""
        return copy.deepcopy(self._training_parameter_binds)

    def bind_training_parameters(
        self, values: Union[Mapping[Parameter, ParameterValueType], Sequence[ParameterValueType]]
    ) -> None:
        """
        Alternate function signature for ``assign_training_parameters``
        """
        self.assign_training_parameters(values)

    def get_unbound_training_parameters(self) -> List[Parameter]:
        """Return a list of any unbound training parameters in the feature map circuit."""
        unbound_training_params = []
        if self._training_parameter_binds is not None:
            # Get all training parameters not associated with numerical values
            unbound_training_params = [
                val
                for val in self._training_parameter_binds.values()
                if not isinstance(val, numbers.Number)
            ]

        return unbound_training_params

    @property  # type: ignore
    @deprecate_property("0.5.0", new_name="training_parameters")
    def user_parameters(self) -> Optional[Union[ParameterVector, Sequence[Parameter]]]:
        """[Deprecated property]Return the vector of training parameters."""
        return self.training_parameters

    @user_parameters.setter  # type: ignore
    @deprecate_property("0.5.0", new_name="training_parameters")
    def user_parameters(self, training_params: Union[ParameterVector, Sequence[Parameter]]) -> None:
        """[Deprecated property setter]Set the training parameters"""
        self.training_parameters = training_params

    @deprecate_method("0.5.0", new_name="assign_training_parameters")
    def assign_user_parameters(
        self, values: Union[Mapping[Parameter, ParameterValueType], Sequence[ParameterValueType]]
    ) -> None:
        """
        [Deprecated method]Assign training parameters in the ``QuantumKernel`` feature map.
        Otherwise, just like ``assign_training_parameters``.

        """
        self.assign_training_parameters(values)

    @property  # type: ignore
    @deprecate_property("0.5.0", new_name="training_parameter_binds")
    def user_param_binds(self) -> Optional[Mapping[Parameter, float]]:
        """
        [Deprecated property]Return a copy of the current training parameter mappings
        for the feature map circuit.
        """
        return self.training_parameter_binds

    @deprecate_method("0.5.0", new_name="bind_training_parameters")
    def bind_user_parameters(
        self, values: Union[Mapping[Parameter, ParameterValueType], Sequence[ParameterValueType]]
    ) -> None:
        """
        [Deprecated method]Alternate function signature for ``assign_training_parameters``
        """
        self.bind_training_parameters(values)

    @deprecate_method("0.5.0", new_name="get_unbound_training_parameters")
    def get_unbound_user_parameters(self) -> List[Parameter]:
        """
        [Deprecated method]Return a list of any unbound training parameters in the feature
        map circuit.
        """
        return self.get_unbound_training_parameters()

    def construct_circuit(
        self,
        x: ParameterVector,
        y: ParameterVector = None,
        measurement: bool = True,
        is_statevector_sim: bool = False,
    ) -> QuantumCircuit:
        r"""
        Construct inner product circuit for given datapoints and feature map.

        If using `statevector_simulator`, only construct circuit for :math:`\Psi(x)|0\rangle`,
        otherwise construct :math:`Psi^dagger(y) x Psi(x)|0>`
        If y is None and not using `statevector_simulator`, self inner product is calculated.

        Args:
            x: first data point parameter vector
            y: second data point parameter vector, ignored if using statevector simulator
            measurement: include measurement if not using statevector simulator
            is_statevector_sim: use state vector simulator

        Returns:
            QuantumCircuit

        Raises:
            ValueError:
                - x and/or y have incompatible dimension with feature map
                - unbound training parameters in the feature map circuit
        """
        qc = self._construct_circuit_with_feature_map(x, y)

        if y is None:
            y = x

        if not is_statevector_sim:
            y_dict = dict(zip(self._feature_map.parameters, y))
            psi_y_dag = self._feature_map.assign_parameters(y_dict)
            qc.append(psi_y_dag.to_instruction().inverse(), qc.qubits)

            if measurement:
                qc.barrier(qc.qregs[0])
                qc.measure(qc.qregs[0], qc.cregs[0])

        return qc

    def _construct_circuit_with_feature_map(self, x: ParameterVector, y: ParameterVector = None):
        self._check_training_parameters_bound()

        self._validate_length(x, "x")
        self._validate_length(y, "y")

        q = QuantumRegister(self._feature_map.num_qubits, "q")
        c = ClassicalRegister(self._feature_map.num_qubits, "c")
        qc = QuantumCircuit(q, c)

        x_dict = dict(zip(self._feature_map.parameters, x))
        psi_x = self._feature_map.assign_parameters(x_dict)
        qc.append(psi_x.to_instruction(), qc.qubits)

        return qc

    # just a synonym
    def _construct_circuit_statevector(self, x: ParameterVector, y: ParameterVector = None):
        """This is just a synonym for ``_construct_circuit_with_feature_map``"""
        return self._construct_circuit_with_feature_map(x, y)

    def _compute_overlap_statevector(self, idx: Tuple[int, int], results: List) -> float:
        """
        Helper function to compute overlap for given input if a statevector simulator is used.
        """
        # |<0|Psi^dagger(y) x Psi(x)|0>|^2, take the amplitude
        v_a, v_b = [results[int(i)] for i in idx]
        tmp = np.vdot(v_a, v_b)
        kernel_value = np.vdot(tmp, tmp).real  # pylint: disable=no-member

        return kernel_value

    def _compute_overlap(self, idx: int, results: Result) -> float:
        """
        Helper function to compute overlap for given input when a non-statevector simulator or
        device is used.
        """
        measurement_basis = "0" * self._feature_map.num_qubits

        result = results.get_counts(idx)
        kernel_value = result.get(measurement_basis, 0) / sum(result.values())

        return kernel_value

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray = None) -> np.ndarray:
        r"""
        Construct kernel matrix for given data and feature map

        If y_vec is None, self inner product is calculated.
        If using `statevector_simulator`, only build circuits for :math:`\Psi(x)|0\rangle`,
        then perform inner product classically.

        Args:
            x_vec: 1D or 2D array of datapoints, NxD, where N is the number of datapoints,
                                                            D is the feature dimension
            y_vec: 1D or 2D array of datapoints, MxD, where M is the number of datapoints,
                                                            D is the feature dimension

        Returns:
            2D matrix, NxM

        Raises:
            QiskitMachineLearningError:
                - A quantum instance or backend has not been provided
            ValueError:
                - unbound training parameters in the feature map circuit
                - x_vec and/or y_vec are not one or two dimensional arrays
                - x_vec and y_vec have have incompatible dimensions
                - x_vec and/or y_vec have incompatible dimension with feature map and
                    and feature map can not be modified to match.
        """
        self._check_training_parameters_bound()

        if self._quantum_instance is None:
            raise QiskitMachineLearningError(
                "A QuantumInstance or Backend must be supplied to evaluate a quantum kernel."
            )

        x_vec, y_vec = self._validate_input(x_vec, y_vec)

        # determine if calculating self inner product
        is_symmetric = True
        if y_vec is None:
            y_vec = x_vec
        elif not np.array_equal(x_vec, y_vec):
            is_symmetric = False

        # initialize kernel matrix
        kernel = np.zeros((x_vec.shape[0], y_vec.shape[0]))

        # set diagonal to 1 if symmetric
        if is_symmetric:
            np.fill_diagonal(kernel, 1)

        is_statevector_sim = self._quantum_instance.is_statevector
        # calculate kernel
        if is_statevector_sim:
            kernel = self._calculate_kernel_statevector(x_vec, y_vec, is_symmetric, kernel)
        else:
            kernel = self._calculate_kernel(x_vec, y_vec, is_symmetric, kernel)

        return kernel

    def _validate_input(self, x_vec: np.ndarray, y_vec: np.ndarray):
        x_vec = np.asarray(x_vec)

        if x_vec.ndim > 2:
            raise ValueError("x_vec must be a 1D or 2D array")

        if x_vec.ndim == 1:
            x_vec = np.reshape(x_vec, (-1, len(x_vec)))

        self._validate_length(x_vec[0, :], "x_vec", adjust_feature_map=True)

        if y_vec is not None:
            y_vec = np.asarray(y_vec)

            if y_vec.ndim == 1:
                y_vec = np.reshape(y_vec, (-1, len(y_vec)))

            if y_vec.ndim > 2:
                raise ValueError("y_vec must be a 1D or 2D array")

            if y_vec.shape[1] != x_vec.shape[1]:
                raise ValueError(
                    "x_vec and y_vec have incompatible dimensions.\n"
                    f"x_vec has {x_vec.shape[1]} dimensions, but y_vec has {y_vec.shape[1]}."
                )

            self._validate_length(y_vec[0, :], "y_vec")

        return x_vec, y_vec

    def _validate_length(
        self, vec: ParameterVector | np.ndarray, vec_name: str, adjust_feature_map: bool = False
    ) -> None:
        """
        Validates the size (length) of the vector against the number of parameters of the feature
        map. Updates the feature map if required to the size of the vector.

        Args:
            vec: A vector to validate.
            vec_name: A vector name, if an error is raised it is used to format the error message.
            adjust_feature_map: If ``True`` tried to adjust the number of qubits of the feature map.

        Raises:
            ValueError: If the size of the vector is not the same as the number of parameters of the
                feature map or when the feature map does not allow to change the number of qubits.
        """
        if vec is not None and len(vec) != self._feature_map.num_parameters:
            raise_error = True
            if adjust_feature_map:
                try:
                    # an attempt to update the feature map with the required number of qubits
                    self._feature_map.num_qubits = vec
                    raise_error = False
                except AttributeError:
                    pass

            if raise_error:
                raise ValueError(
                    f"{vec_name} and class feature map have incompatible dimensions.\n"
                    f"{vec_name} has {len(vec)} dimensions, "
                    f"but feature map has {self._feature_map.num_parameters}."
                )

    def _calculate_kernel_statevector(
        self, x_vec: np.ndarray, y_vec: np.ndarray, is_symmetric: bool, kernel: np.ndarray
    ):
        # get indices to calculate
        row_indices, col_indices = self._get_indices(x_vec, y_vec, is_symmetric)

        if is_symmetric:
            to_be_computed_data = x_vec
        else:  # not symmetric
            to_be_computed_data = np.concatenate((x_vec, y_vec))

        feature_map_params = ParameterVector("par_x", self._feature_map.num_parameters)
        parameterized_circuit = self._construct_circuit_statevector(
            feature_map_params,
            feature_map_params,
        )
        parameterized_circuit = self._quantum_instance.transpile(
            parameterized_circuit, pass_manager=self._quantum_instance.unbound_pass_manager
        )[0]
        statevectors = []

        for min_idx in range(0, len(to_be_computed_data), self._batch_size):
            max_idx = min(min_idx + self._batch_size, len(to_be_computed_data))
            circuits = [
                parameterized_circuit.assign_parameters({feature_map_params: x})
                for x in to_be_computed_data[min_idx:max_idx]
            ]
            if self._quantum_instance.bound_pass_manager is not None:
                circuits = self._quantum_instance.transpile(
                    circuits, pass_manager=self._quantum_instance.bound_pass_manager
                )
            results = self._quantum_instance.execute(circuits, had_transpiled=True)
            for j in range(max_idx - min_idx):
                statevectors.append(results.get_statevector(j))

        offset = 0 if is_symmetric else len(x_vec)
        matrix_elements = [
            self._compute_overlap_statevector(idx, statevectors)
            for idx in list(zip(row_indices, col_indices + offset))
        ]

        for i, j, value in zip(row_indices, col_indices, matrix_elements):
            kernel[i, j] = value
            if is_symmetric:
                kernel[j, i] = kernel[i, j]

        return kernel

    def _calculate_kernel(
        self, x_vec: np.ndarray, y_vec: np.ndarray, is_symmetric: bool, kernel: np.ndarray
    ):
        # get indices to calculate
        row_indices, col_indices = self._get_indices(x_vec, y_vec, is_symmetric)

        feature_map_params_x = ParameterVector("par_x", self._feature_map.num_parameters)
        feature_map_params_y = ParameterVector("par_y", self._feature_map.num_parameters)
        parameterized_circuit = self.construct_circuit(feature_map_params_x, feature_map_params_y)
        parameterized_circuit = self._quantum_instance.transpile(
            parameterized_circuit, pass_manager=self._quantum_instance.unbound_pass_manager
        )[0]

        for idx in range(0, len(row_indices), self._batch_size):
            to_be_computed_data_pair = []
            to_be_computed_index = []
            for sub_idx in range(idx, min(idx + self._batch_size, len(row_indices))):
                i = row_indices[sub_idx]
                j = col_indices[sub_idx]
                x_i = x_vec[i]
                y_j = y_vec[j]
                if not np.all(x_i == y_j):
                    to_be_computed_data_pair.append((x_i, y_j))
                    to_be_computed_index.append((i, j))

            circuits = [
                parameterized_circuit.assign_parameters(
                    {feature_map_params_x: x, feature_map_params_y: y}
                )
                for x, y in to_be_computed_data_pair
            ]
            if self._quantum_instance.bound_pass_manager is not None:
                circuits = self._quantum_instance.transpile(
                    circuits, pass_manager=self._quantum_instance.bound_pass_manager
                )

            results = self._quantum_instance.execute(circuits, had_transpiled=True)

            matrix_elements = [
                self._compute_overlap(circuit, results) for circuit in range(len(circuits))
            ]

            for (i, j), value in zip(to_be_computed_index, matrix_elements):
                kernel[i, j] = value
                if is_symmetric:
                    kernel[j, i] = kernel[i, j]

        if self._enforce_psd and is_symmetric:
            # Find the closest positive semi-definite approximation to symmetric kernel matrix.
            # The (symmetric) matrix should always be positive semi-definite by construction,
            # but this can be violated in case of noise, such as sampling noise, thus the
            # adjustment is only done if NOT using the statevector simulation.
            D, U = np.linalg.eig(kernel)  # pylint: disable=invalid-name
            kernel = U @ np.diag(np.maximum(0, D)) @ U.transpose()

        return kernel

    def _get_indices(self, x_vec: np.ndarray, y_vec: np.ndarray, is_symmetric: bool):
        # get indices to calculate
        if is_symmetric:
            row_indices, col_indices = np.triu_indices(x_vec.shape[0], k=1)  # remove diagonal
        else:
            row_indices, col_indices = np.indices((x_vec.shape[0], y_vec.shape[0]))
            row_indices = np.asarray(row_indices.flat)
            col_indices = np.asarray(col_indices.flat)
        return row_indices, col_indices

    def _check_training_parameters_bound(self):
        # Ensure all training parameters have been bound in the feature map circuit.
        unbound_params = self.get_unbound_training_parameters()
        if unbound_params:
            raise ValueError(
                f"""
                The feature map circuit contains unbound training parameters ({unbound_params}).
                All training parameters must be bound to numerical values before evaluating
                the kernel matrix.
                """
            )
