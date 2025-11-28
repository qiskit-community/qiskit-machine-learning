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
A class for the Quantum Fisher Information.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence

from qiskit.circuit import Parameter, QuantumCircuit

from .base.base_qgt import BaseQGT
from .lin_comb.lin_comb_estimator_gradient import DerivativeType
from .qfi_result import QFIResult
from ..algorithm_job import AlgorithmJob
from ..exceptions import AlgorithmError


class QFI(ABC):
    r"""Computes the Quantum Fisher Information (QFI) given a pure,
    parameterized quantum state. QFI is defined as:

    .. math::

        \mathrm{QFI}_{ij}= 4 \mathrm{Re}[\langle \partial_i \psi | \partial_j \psi \rangle
        - \langle\partial_i \psi | \psi \rangle \langle\psi | \partial_j \psi \rangle].
    """

    def __init__(
        self,
        qgt: BaseQGT,
        precision: float | None = None,
    ):
        r"""
        Args:
            qgt: The quantum geometric tensor used to compute the QFI.
            precision: Precision to override the BaseQGT's. If None, the BaseQGT's precision will
                be used.
        """
        self._qgt: BaseQGT = qgt
        self._precision = precision

    def run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None = None,
        *,
        precision: float | Sequence[float] | None = None,
    ) -> AlgorithmJob:
        """Run the job of the QFIs on the given circuits.

        Args:
            circuits: The list of quantum circuits to compute the QFIs.
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The sequence of parameters to calculate only the QFIs of
                the specified parameters. Each sequence of parameters corresponds to a circuit in
                ``circuits``. Defaults to None, which means that the QFIs of all parameters in
                each circuit are calculated.
            precision: Precision to be used by the underlying Estimator. If a single float is
                provided, this number will be used for all circuits. If a sequence of floats is
                provided, they will be used on a per-circuit basis. If not set, the gradient's default
                precision will be used for all circuits, and if that is None (not set) then the
                underlying primitive's (default) precision will be used for all circuits.

        Returns:
            The job object of the QFIs of the expectation values. The i-th result corresponds to
            ``circuits[i]`` evaluated with parameters bound as ``parameter_values[i]``.
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

        if precision is None:
            precision = self.precision  # May still be None

        job = AlgorithmJob(self._run, circuits, parameter_values, parameters, precision=precision)
        job._submit()
        return job

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        *,
        precision: float | Sequence[float] | None,
    ) -> QFIResult:
        """Compute the QFI on the given circuits."""
        # Set the derivative type to real
        temp_derivative_type, self._qgt.derivative_type = (
            self._qgt.derivative_type,
            DerivativeType.REAL,
        )

        job = self._qgt.run(circuits, parameter_values, parameters, precision=precision)

        try:
            result = job.result()
        except AlgorithmError as exc:
            raise AlgorithmError("Estimator job or gradient job failed.") from exc

        self._qgt.derivative_type = temp_derivative_type

        return QFIResult(
            qfis=[4 * qgt.real for qgt in result.qgts],
            metadata=result.metadata,
            precision=result.precision,
        )

    @property
    def precision(self) -> float | None:
        """Return the precision used by the `run` method of the BaseQGT's Estimator primitive. If
        None, the default precision of the primitive is used.

        Returns:
            The default precision.
        """
        return self._precision

    @precision.setter
    def precision(self, precision: float | None):
        """Update the QFI's default precision setting.

        Args:
            precision: The new default precision.
        """

        self._precision = precision
