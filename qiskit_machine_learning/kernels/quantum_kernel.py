# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Kernel Algorithm"""

from typing import Optional, Union, Sequence, Mapping, List
import copy
import numbers

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.library import ZZFeatureMap
from qiskit.providers import Backend, BaseBackend
from qiskit.utils import QuantumInstance
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

    def __init__(
        self,
        feature_map: Optional[QuantumCircuit] = None,
        enforce_psd: bool = True,
        batch_size: int = 900,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
        user_parameters: Optional[Union[ParameterVector, Sequence[Parameter]]] = None,
    ) -> None:
        """
        Args:
            feature_map: Parameterized circuit to be used as the feature map. If None is given,
                the `ZZFeatureMap` is used with two qubits.
            enforce_psd: Project to closest positive semidefinite matrix if x = y.
                Only enforced when not using the state vector simulator. Default True.
            batch_size: Number of circuits to batch together for computation. Default 1000.
            quantum_instance: Quantum Instance or Backend
            user_parameters: Iterable containing ``Parameter`` objects which correspond to
                 quantum gates on the feature map circuit which may be tuned. If users intend to
                 tune feature map parameters to find optimal values, this field should be set.
        """
        # Class fields
        self._feature_map = None
        self._unbound_feature_map = None
        self._user_parameters = None
        self._user_param_binds = None
        self._enforce_psd = enforce_psd
        self._batch_size = batch_size
        self._quantum_instance = quantum_instance

        # Setters
        self.feature_map = feature_map if feature_map else ZZFeatureMap(2)
        if user_parameters is not None:
            self.user_parameters = user_parameters

    @property
    def feature_map(self) -> QuantumCircuit:
        """Returns feature map"""
        return self._feature_map

    @feature_map.setter
    def feature_map(self, feature_map: QuantumCircuit) -> None:
        """
        Sets feature map.

        The ``unbound_feature_map`` field will be automatically updated when this field is set,
        and ``user_parameters`` and ``user_param_binds`` fields will be reset to ``None``.
        """
        self._feature_map = feature_map
        self._unbound_feature_map = copy.deepcopy(self._feature_map)
        self._user_parameters = None
        self._user_param_binds = None

    @property
    def unbound_feature_map(self) -> QuantumCircuit:
        """Returns unbound feature map"""
        return copy.deepcopy(self._unbound_feature_map)

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Returns quantum instance"""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(
        self, quantum_instance: Union[Backend, BaseBackend, QuantumInstance]
    ) -> None:
        """Sets quantum instance"""
        if isinstance(quantum_instance, (BaseBackend, Backend)):
            self._quantum_instance = QuantumInstance(quantum_instance)
        else:
            self._quantum_instance = quantum_instance

    @property
    def user_parameters(self) -> Optional[Union[ParameterVector, Sequence[Parameter]]]:
        """Return the vector of user parameters."""
        return self._user_parameters

    @user_parameters.setter
    def user_parameters(self, user_params: Union[ParameterVector, Sequence[Parameter]]) -> None:
        """Sets the user parameters"""
        self._user_param_binds = {user_params[i]: user_params[i] for i, _ in enumerate(user_params)}

        self._user_parameters = user_params

    def assign_user_parameters(
        self, values: Union[Mapping[Parameter, ParameterValueType], Sequence[ParameterValueType]]
    ) -> None:
        """
        Assign user parameters in the ``QuantumKernel`` feature map.

        Args:
            values (dict or iterable): Either a dictionary or iterable specifying the new
                parameter values. If a dict, it specifies the mapping from ``current_parameter`` to
                ``new_parameter``, where ``new_parameter`` can be a parameter object or a
                numeric value. If an iterable, the elements are assigned to the existing parameters
                in the order of ``QuantumKernel.user_parameters``.

        Raises:
            ValueError: Incompatible number of user parameters and values
        """
        if self._user_parameters is None:
            raise ValueError(
                """
                The number of parameter values ({len(values)}) does not
                match the number of user parameters tracked by the QuantumKernel
                (None).
                """
            )

        if isinstance(values, dict):
            unknown_parameters = list(set(values.keys()) - set(self._user_parameters))
            if len(unknown_parameters) > 0:
                raise ValueError(
                    f"Cannot bind parameters ({unknown_parameters}) not tracked by the quantum kernel."
                )
            param_binds = values
        else:
            if len(values) != len(self._user_parameters):
                raise ValueError(
                    f"""
                The number of parameter values ({len(values)}) does not
                match the number of user parameters tracked by the QuantumKernel
                ({len(self._user_parameters)}).
                """
                )
            param_binds = {param: values[i] for i, param in enumerate(self._user_parameters)}

        if self._user_param_binds is None:
            self._user_param_binds = param_binds
        else:
            self._user_param_binds.update(param_binds)
        self._feature_map = self._unbound_feature_map.assign_parameters(self._user_param_binds)

    @property
    def user_param_binds(self) -> Optional[Mapping[Parameter, float]]:
        """Return a copy of the current user parameter mappings for the feature map circuit."""
        return copy.deepcopy(self._user_param_binds)

    def bind_user_parameters(self, values: Sequence[float]) -> None:
        """
        Alternate function signature for ``assign_user_parameters``

        Args:
            values (iterable): [value1, value2, ...]
        """
        self.assign_user_parameters(values)

    def get_unbound_parameters(self) -> List[Parameter]:
        """Returns a list of any unbound user parameters in the feature map circuit."""
        unbound_user_params = []
        if self._user_param_binds is not None:
            # Get all user parameters not associated with numerical values
            unbound_user_params = [
                val
                for val in self._user_param_binds.values()
                if not isinstance(val, numbers.Number)
            ]

        return unbound_user_params

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
                - unbound user parameters in the feature map circuit
        """
        # Ensure all user parameters have been bound in the feature map circuit.
        unbound_params = self.get_unbound_parameters()
        if unbound_params:
            raise ValueError(
                f"""
                The feature map circuit contains unbound user parameters ({unbound_params}).
                All user parameters must be bound to numerical values before constructing
                inner product circuit.
                """
            )

        if len(x) != self._feature_map.num_parameters:
            raise ValueError(
                "x and class feature map incompatible dimensions.\n"
                f"x has {len(x)} dimensions, but feature map has {self._feature_map.num_parameters}."
            )

        q = QuantumRegister(self._feature_map.num_qubits, "q")
        c = ClassicalRegister(self._feature_map.num_qubits, "c")
        qc = QuantumCircuit(q, c)

        x_dict = dict(zip(self._feature_map.parameters, x))
        psi_x = self._feature_map.assign_parameters(x_dict)
        qc.append(psi_x.to_instruction(), qc.qubits)

        if not is_statevector_sim:
            if y is not None and len(y) != self._feature_map.num_parameters:
                raise ValueError(
                    "y and class feature map incompatible dimensions.\n"
                    f"y has {len(y)} dimensions, but feature map has {self._feature_map.num_parameters}."
                )

            if y is None:
                y = x

            y_dict = dict(zip(self._feature_map.parameters, y))
            psi_y_dag = self._feature_map.assign_parameters(y_dict)
            qc.append(psi_y_dag.to_instruction().inverse(), qc.qubits)

            if measurement:
                qc.barrier(q)
                qc.measure(q, c)
        return qc

    def _compute_overlap(self, idx, results, is_statevector_sim, measurement_basis) -> float:
        """
        Helper function to compute overlap for given input.
        """
        if is_statevector_sim:
            # |<0|Psi^dagger(y) x Psi(x)|0>|^2, take the amplitude
            v_a, v_b = [results.get_statevector(int(i)) for i in idx]
            tmp = np.vdot(v_a, v_b)
            kernel_value = np.vdot(tmp, tmp).real  # pylint: disable=no-member
        else:
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
                - unbound user parameters in the feature map circuit
                - x_vec and/or y_vec are not one or two dimensional arrays
                - x_vec and y_vec have have incompatible dimensions
                - x_vec and/or y_vec have incompatible dimension with feature map and
                    and feature map can not be modified to match.
        """
        # Ensure all user parameters have been bound in the feature map circuit.
        unbound_params = self.get_unbound_parameters()
        if unbound_params:
            raise ValueError(
                f"""
                The feature map circuit contains unbound user parameters ({unbound_params}).
                All user parameters must be bound to numerical values before evaluating
                the kernel matrix.
                """
            )

        if self._quantum_instance is None:
            raise QiskitMachineLearningError(
                "A QuantumInstance or Backend must be supplied to evaluate a quantum kernel."
            )
        if isinstance(self._quantum_instance, (BaseBackend, Backend)):
            self._quantum_instance = QuantumInstance(self._quantum_instance)

        if not isinstance(x_vec, np.ndarray):
            x_vec = np.asarray(x_vec)
        if y_vec is not None and not isinstance(y_vec, np.ndarray):
            y_vec = np.asarray(y_vec)

        if x_vec.ndim > 2:
            raise ValueError("x_vec must be a 1D or 2D array")

        if x_vec.ndim == 1:
            x_vec = np.reshape(x_vec, (-1, 2))

        if y_vec is not None and y_vec.ndim > 2:
            raise ValueError("y_vec must be a 1D or 2D array")

        if y_vec is not None and y_vec.ndim == 1:
            y_vec = np.reshape(y_vec, (-1, 2))

        if y_vec is not None and y_vec.shape[1] != x_vec.shape[1]:
            raise ValueError(
                "x_vec and y_vec have incompatible dimensions.\n"
                f"x_vec has {x_vec.shape[1]} dimensions, but y_vec has {y_vec.shape[1]}."
            )

        if x_vec.shape[1] != self._feature_map.num_parameters:
            try:
                self._feature_map.num_qubits = x_vec.shape[1]
            except AttributeError:
                raise ValueError(
                    "x_vec and class feature map have incompatible dimensions.\n"
                    f"x_vec has {x_vec.shape[1]} dimensions, "
                    f"but feature map has {self._feature_map.num_parameters}."
                ) from AttributeError

        if y_vec is not None and y_vec.shape[1] != self._feature_map.num_parameters:
            raise ValueError(
                "y_vec and class feature map have incompatible dimensions.\n"
                f"y_vec has {y_vec.shape[1]} dimensions, but feature map "
                f"has {self._feature_map.num_parameters}."
            )

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

        # get indices to calculate
        if is_symmetric:
            mus, nus = np.triu_indices(x_vec.shape[0], k=1)  # remove diagonal
        else:
            mus, nus = np.indices((x_vec.shape[0], y_vec.shape[0]))
            mus = np.asarray(mus.flat)
            nus = np.asarray(nus.flat)

        is_statevector_sim = self._quantum_instance.is_statevector
        measurement = not is_statevector_sim
        measurement_basis = "0" * self._feature_map.num_qubits

        # calculate kernel
        if is_statevector_sim:  # using state vector simulator
            if is_symmetric:
                to_be_computed_data = x_vec
            else:  # not symmetric
                to_be_computed_data = np.concatenate((x_vec, y_vec))

            feature_map_params = ParameterVector("par_x", self._feature_map.num_parameters)
            parameterized_circuit = self.construct_circuit(
                feature_map_params,
                feature_map_params,
                measurement=measurement,
                is_statevector_sim=is_statevector_sim,
            )
            parameterized_circuit = self._quantum_instance.transpile(parameterized_circuit)[0]
            circuits = [
                parameterized_circuit.assign_parameters({feature_map_params: x})
                for x in to_be_computed_data
            ]

            results = self._quantum_instance.execute(circuits)

            offset = 0 if is_symmetric else len(x_vec)
            matrix_elements = [
                self._compute_overlap(idx, results, is_statevector_sim, measurement_basis)
                for idx in list(zip(mus, nus + offset))
            ]

            for i, j, value in zip(mus, nus, matrix_elements):
                kernel[i, j] = value
                if is_symmetric:
                    kernel[j, i] = kernel[i, j]

        else:  # not using state vector simulator
            feature_map_params_x = ParameterVector("par_x", self._feature_map.num_parameters)
            feature_map_params_y = ParameterVector("par_y", self._feature_map.num_parameters)
            parameterized_circuit = self.construct_circuit(
                feature_map_params_x,
                feature_map_params_y,
                measurement=measurement,
                is_statevector_sim=is_statevector_sim,
            )
            parameterized_circuit = self._quantum_instance.transpile(parameterized_circuit)[0]

            for idx in range(0, len(mus), self._batch_size):
                to_be_computed_data_pair = []
                to_be_computed_index = []
                for sub_idx in range(idx, min(idx + self._batch_size, len(mus))):
                    i = mus[sub_idx]
                    j = nus[sub_idx]
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

                results = self._quantum_instance.execute(circuits)

                matrix_elements = [
                    self._compute_overlap(circuit, results, is_statevector_sim, measurement_basis)
                    for circuit in range(len(circuits))
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
