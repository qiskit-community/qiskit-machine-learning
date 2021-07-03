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

from typing import Optional, Union

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
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
        batch_size: int = 1000,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
    ) -> None:
        """
        Args:
            feature_map: Parameterized circuit to be used as the feature map. If None is given,
                the `ZZFeatureMap` is used with two qubits.
            enforce_psd: Project to closest positive semidefinite matrix if x = y.
                Only enforced when not using the state vector simulator. Default True.
            batch_size: Number of circuits to batch together for computation. Default 1000.
            quantum_instance: Quantum Instance or Backend
        """
        self._feature_map = feature_map if feature_map else ZZFeatureMap(2)
        self._enforce_psd = enforce_psd
        self._batch_size = batch_size
        self._quantum_instance = quantum_instance

    @property
    def feature_map(self) -> QuantumCircuit:
        """Returns feature map"""
        return self._feature_map

    @feature_map.setter
    def feature_map(self, feature_map: QuantumCircuit):
        """Sets feature map"""
        self._feature_map = feature_map

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
        """

        if len(x) != self._feature_map.num_parameters:
            raise ValueError(
                "x and class feature map incompatible dimensions.\n"
                + "x has %s dimensions, but feature map has %s."
                % (len(x), self._feature_map.num_parameters)
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
                    + "y has %s dimensions, but feature map has %s."
                    % (len(y), self._feature_map.num_parameters)
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

    def _compute_overlap(self, idx, results, is_statevector_sim, measurement_basis):
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
                - x_vec and/or y_vec are not one or two dimensional arrays
                - x_vec and y_vec have have incompatible dimensions
                - x_vec and/or y_vec have incompatible dimension with feature map and
                    and feature map can not be modified to match.
        """
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
                + "x_vec has %s dimensions, but y_vec has %s." % (x_vec.shape[1], y_vec.shape[1])
            )

        if x_vec.shape[1] != self._feature_map.num_parameters:
            try:
                self._feature_map.num_qubits = x_vec.shape[1]
            except AttributeError:
                raise ValueError(
                    "x_vec and class feature map have incompatible dimensions.\n"
                    + "x_vec has %s dimensions, but feature map has %s."
                    % (x_vec.shape[1], self._feature_map.num_parameters)
                ) from AttributeError

        if y_vec is not None and y_vec.shape[1] != self._feature_map.num_parameters:
            raise ValueError(
                "y_vec and class feature map have incompatible dimensions.\n"
                + "y_vec has %s dimensions, but feature map has %s."
                % (y_vec.shape[1], self._feature_map.num_parameters)
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
