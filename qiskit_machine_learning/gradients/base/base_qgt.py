# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Abstract base class of the Quantum Geometric Tensor (QGT).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.primitives import BaseEstimatorV2
from qiskit.transpiler.passes import TranslateParameterizedGates
from qiskit.passmanager import BasePassManager
from ...utils import circuit_cache_key

from .qgt_result import QGTResult
from ..utils import (
    DerivativeType,
    GradientCircuit,
    _assign_unique_parameters,
    _make_gradient_parameters,
    _make_gradient_parameter_values,
)

from ...algorithm_job import AlgorithmJob


class BaseQGT(ABC):
    r"""Base class to computes the Quantum Geometric Tensor (QGT) given a pure,
    parameterized quantum state. QGT is defined as:

    .. math::

        \mathrm{QGT}_{ij}= \langle \partial_i \psi | \partial_j \psi \rangle
            - \langle\partial_i \psi | \psi \rangle \langle\psi | \partial_j \psi \rangle.
    """

    def __init__(
        self,
        estimator: BaseEstimatorV2,
        phase_fix: bool = True,
        derivative_type: DerivativeType = DerivativeType.COMPLEX,
        precision: float | None = None,
        *,
        pass_manager: BasePassManager | None = None,
        pass_manager_options: dict[str, Any] | None = None,
    ):
        r"""
        Args:
            estimator: The estimator used to compute the QGT.
            phase_fix: Whether to calculate the second term (phase fix) of the QGT, which is
                :math:`\langle\partial_i \psi | \psi \rangle \langle\psi | \partial_j \psi \rangle`.
                Defaults to ``True``.
            derivative_type: The type of derivative. Can be either ``DerivativeType.REAL``
                ``DerivativeType.IMAG``, or ``DerivativeType.COMPLEX``. Defaults to
                ``DerivativeType.REAL``.

                - ``DerivativeType.REAL`` computes

                .. math::

                    \mathrm{Re(QGT)}_{ij}= \mathrm{Re}[\langle \partial_i \psi | \partial_j \psi \rangle
                        - \langle\partial_i \psi | \psi \rangle \langle\psi | \partial_j \psi \rangle].

                - ``DerivativeType.IMAG`` computes

                .. math::

                    \mathrm{Im(QGT)}_{ij}= \mathrm{Im}[\langle \partial_i \psi | \partial_j \psi \rangle
                        - \langle\partial_i \psi | \psi \rangle \langle\psi | \partial_j \psi \rangle].

                - ``DerivativeType.COMPLEX`` computes

                .. math::

                    \mathrm{QGT}_{ij}= [\langle \partial_i \psi | \partial_j \psi \rangle
                        - \langle\partial_i \psi | \psi \rangle \langle\psi | \partial_j \psi \rangle].
            precision: Precision to be used by the underlying Estimator. If provided, this number
                takes precedence over the default precision of the primitive. If None, the default
                precision of the primitive is used.
            pass_manager: An optional object with a `run` method allowing to transpile the circuits
                that are run when using this algorithm. If set to `None`, these won't be
                transpiled.
        """
        self._estimator: BaseEstimatorV2 = estimator
        self._precision = precision
        self._phase_fix: bool = phase_fix
        self._derivative_type: DerivativeType = derivative_type
        self._qgt_circuit_cache: dict[tuple, GradientCircuit] = {}
        self._gradient_circuit_cache: dict[
            str | tuple,
            GradientCircuit,
        ] = {}

        self._pass_manager = pass_manager
        self._pass_manager_options = (
            pass_manager_options if pass_manager_options is not None else {}
        )

    @property
    def derivative_type(self) -> DerivativeType:
        """The derivative type."""
        return self._derivative_type

    @derivative_type.setter
    def derivative_type(self, derivative_type: DerivativeType) -> None:
        """Set the derivative type."""
        self._derivative_type = derivative_type

    def run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None = None,
        *,
        precision: float | Sequence[float] | None = None,
    ) -> AlgorithmJob:
        """Run the job of the QGTs on the given circuits.

        Args:
            circuits: The list of quantum circuits to compute the QGTs.
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The sequence of parameters to calculate only the QGTs of
                the specified parameters. Each sequence of parameters corresponds to a circuit in
                ``circuits``. Defaults to None, which means that the QGTs of all parameters in
                each circuit are calculated.
            precision: Precision to be used by the underlying Estimator. If a single float is
                provided, this number will be used for all circuits. If a sequence of floats is
                provided, they will be used on a per-circuit basis. If not set, the gradient's default
                precision will be used for all circuits, and if that is None (not set) then the
                underlying primitive's (default) precision will be used for all circuits.

        Returns:
            The job object of the QGTs of the expectation values. The i-th result corresponds to
            ``circuits[i]`` evaluated with parameters bound as ``parameter_values[i]``.

        Raises:
            ValueError: Invalid arguments are given.
        """
        if isinstance(circuits, QuantumCircuit):
            # Allow a single circuit to be passed in.
            circuits = (circuits,)

        if parameters is None:
            # If parameters is None, we calculate the gradients of all parameters in each circuit.
            parameters = [circuit.parameters for circuit in circuits]
        else:
            # If parameters is not None, we calculate the gradients of the specified parameters.
            # None in parameters means that the gradients of all parameters in the corresponding
            # circuit are calculated.
            parameters = [
                params if params is not None else circuits[i].parameters
                for i, params in enumerate(parameters)
            ]
        # Validate the arguments.
        self._validate_arguments(circuits, parameter_values, parameters)

        if precision is None:
            precision = self.precision  # May still be None

        job = AlgorithmJob(self._run, circuits, parameter_values, parameters, precision=precision)
        job._submit()
        return job

    @abstractmethod
    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        *,
        precision: float | Sequence[float] | None,
    ) -> QGTResult:
        """Compute the QGTs on the given circuits."""
        raise NotImplementedError()

    def _preprocess(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        supported_gates: Sequence[str],
    ) -> tuple[Sequence[QuantumCircuit], Sequence[Sequence[float]], Sequence[Sequence[Parameter]]]:
        """Preprocess the gradient. This makes a gradient circuit for each circuit. The gradient
        circuit is a transpiled circuit by using the supported gates, and has unique parameters.
        ``parameter_values`` and ``parameters`` are also updated to match the gradient circuit.

        Args:
            circuits: The list of quantum circuits to compute the gradients.
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The sequence of parameters to calculate only the gradients of the specified
                parameters.
            supported_gates: The supported gates used to transpile the circuit.

        Returns:
            The list of gradient circuits, the list of parameter values, and the list of parameters.
            parameter_values and parameters are updated to match the gradient circuit.
        """
        translator = TranslateParameterizedGates(supported_gates)
        g_circuits: list[QuantumCircuit] = []
        g_parameter_values: list[Sequence[float]] = []
        g_parameters: list[Sequence[Parameter]] = []
        for circuit, parameter_value_, parameters_ in zip(circuits, parameter_values, parameters):
            circuit_key = circuit_cache_key(circuit)
            if circuit_key not in self._gradient_circuit_cache:
                unrolled = translator(circuit)
                self._gradient_circuit_cache[circuit_key] = _assign_unique_parameters(unrolled)
            gradient_circuit = self._gradient_circuit_cache[circuit_key]
            g_circuits.append(gradient_circuit.gradient_circuit)
            g_parameter_values.append(
                _make_gradient_parameter_values(  # type: ignore[arg-type]
                    circuit, gradient_circuit, parameter_value_
                )
            )
            g_parameters_ = [
                g_param
                for g_param in gradient_circuit.gradient_circuit.parameters
                if g_param in _make_gradient_parameters(gradient_circuit, parameters_)
            ]
            g_parameters.append(g_parameters_)
        return g_circuits, g_parameter_values, g_parameters

    def _postprocess(
        self,
        results: QGTResult,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
    ) -> QGTResult:
        """Postprocess the QGTs. This method computes the QGTs of the original circuits
        by applying the chain rule to the QGTs of the circuits with unique parameters.

        Args:
            results: The computed QGT for the circuits with unique parameters.
            circuits: The list of original circuits submitted for gradient computation.
            parameter_values: The list of parameter values to be bound to the circuits.
            parameters: The sequence of parameters to calculate only the gradients of the specified
                parameters.

        Returns:
            The QGTs of the original circuits.
        """
        qgts, metadata = [], []
        for idx, (circuit, parameter_values_, parameters_) in enumerate(
            zip(circuits, parameter_values, parameters)
        ):
            dtype = complex if self.derivative_type == DerivativeType.COMPLEX else float
            qgt: np.ndarray = np.zeros((len(parameters_), len(parameters_)), dtype=dtype)

            gradient_circuit = self._gradient_circuit_cache[circuit_cache_key(circuit)]
            g_parameters = _make_gradient_parameters(gradient_circuit, parameters_)
            # Make a map from the gradient parameter to the respective index in the gradient.
            # parameters_ = [param for param in circuit.parameters if param in parameters_]
            g_parameter_indices = [
                param
                for param in gradient_circuit.gradient_circuit.parameters
                if param in g_parameters
            ]
            g_parameter_indices_d = {param: i for i, param in enumerate(g_parameter_indices)}
            rows, cols = np.triu_indices(len(parameters_))
            for row, col in zip(rows, cols):
                for g_parameter1, coeff1 in gradient_circuit.parameter_map[parameters_[row]]:
                    for g_parameter2, coeff2 in gradient_circuit.parameter_map[parameters_[col]]:
                        if isinstance(coeff1, ParameterExpression):
                            local_map = {
                                p: parameter_values_[circuit.parameters.data.index(p)]
                                for p in coeff1.parameters
                            }
                            bound_coeff1 = coeff1.bind(local_map)
                        else:
                            bound_coeff1 = coeff1
                        if isinstance(coeff2, ParameterExpression):
                            local_map = {
                                p: parameter_values_[circuit.parameters.data.index(p)]
                                for p in coeff2.parameters
                            }
                            bound_coeff2 = coeff2.bind(local_map)
                        else:
                            bound_coeff2 = coeff2
                        qgt[row, col] += (
                            float(bound_coeff1)
                            * float(bound_coeff2)
                            * results.qgts[idx][
                                g_parameter_indices_d[g_parameter1],
                                g_parameter_indices_d[g_parameter2],
                            ]
                        )

            if self.derivative_type == DerivativeType.IMAG:
                qgt += -1 * np.triu(qgt, k=1).T
            else:
                qgt += np.triu(qgt, k=1).conjugate().T
            qgts.append(qgt)
            metadata.append([{"parameters": parameters_}])
        return QGTResult(
            qgts=qgts,
            derivative_type=self.derivative_type,
            metadata=metadata,
            precision=results.precision,
        )

    @staticmethod
    def _validate_arguments(
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
    ) -> None:
        """Validate the arguments of the ``run`` method.

        Args:
            circuits: The list of quantum circuits to compute the QGTs.
            parameter_values: The list of parameter values to be bound to the circuits.
            parameters: The sequence of parameters with respect to which the QGTs should be
                computed.

        Raises:
            ValueError: Invalid arguments are given.
        """
        if len(circuits) != len(parameter_values):
            raise ValueError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of parameter values ({len(parameter_values)})."
            )

        if len(circuits) != len(parameters):
            raise ValueError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of the specified parameter sets ({len(parameters)})."
            )

        for i, (circuit, parameter_value) in enumerate(zip(circuits, parameter_values)):
            if not circuit.num_parameters:
                raise ValueError(f"The {i}-th circuit is not parameterised.")
            if len(parameter_value) != circuit.num_parameters:
                raise ValueError(
                    f"The number of values ({len(parameter_value)}) does not match "
                    f"the number of parameters ({circuit.num_parameters}) for the {i}-th circuit."
                )

        if len(circuits) != len(parameters):
            raise ValueError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of the list of specified parameters ({len(parameters)})."
            )

        for i, (circuit, parameters_) in enumerate(zip(circuits, parameters)):
            if not set(parameters_).issubset(circuit.parameters):
                raise ValueError(
                    f"The {i}-th parameters contains parameters not present in the "
                    f"{i}-th circuit."
                )

    @property
    def precision(self) -> float | None:
        """Return the precision used by the `run` method of the Estimator primitive. If None,
        the default precision of the primitive is used.

        Returns:
            The default precision.
        """
        return self._precision

    @precision.setter
    def precision(self, precision: float | None):
        """Update the gradient's default precision setting.

        Args:
            precision: The new default precision.
        """

        self._precision = precision
