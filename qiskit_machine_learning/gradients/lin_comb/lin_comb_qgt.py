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
A class for the Linear Combination Quantum Gradient Tensor.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.passmanager import BasePassManager

from ...utils import circuit_cache_key
from ..base.base_qgt import BaseQGT
from .lin_comb_estimator_gradient import LinCombEstimatorGradient
from ..base.qgt_result import QGTResult
from ..utils import DerivativeType, _make_lin_comb_qgt_circuit, _make_lin_comb_observables

from ...exceptions import AlgorithmError


class LinCombQGT(BaseQGT):
    """Computes the Quantum Geometric Tensor (QGT) given a pure, parameterized quantum state.

    This method employs a linear combination of unitaries [1].

    **Reference:**

        [1]: Schuld et al., "Evaluating analytic gradients on quantum hardware" (2018).
             `arXiv:1811.11184 <https://arxiv.org/pdf/1811.11184.pdf>`_
    """

    SUPPORTED_GATES = [
        "rx",
        "ry",
        "rz",
        "rzx",
        "rzz",
        "ryy",
        "rxx",
        "cx",
        "cy",
        "cz",
        "ccx",
        "swap",
        "iswap",
        "h",
        "t",
        "s",
        "sdg",
        "x",
        "y",
        "z",
    ]

    def __init__(
        self,
        estimator: BaseEstimatorV2,
        phase_fix: bool = True,
        derivative_type: DerivativeType = DerivativeType.COMPLEX,
        *,
        pass_manager: BasePassManager | None = None,
    ):
        r"""
        Args:
            estimator: The estimator used to compute the QGT.
            phase_fix: Whether to calculate the second term (phase fix) of the QGT, which is
                :math:`\langle\partial_i \psi | \psi \rangle \langle\psi | \partial_j \psi \rangle`.
                Default to ``True``.
            derivative_type: The type of derivative. Can be either ``DerivativeType.REAL``
                ``DerivativeType.IMAG``, or ``DerivativeType.COMPLEX``. Defaults to
                ``DerivativeType.REAL``.

                - ``DerivativeType.REAL`` computes

                .. math::

                    \mathrm{Re(QGT)}_{ij}= \mathrm{Re}[\langle \partial_i \psi | \partial_j \psi \rangle
                        - \langle\partial_i \psi | \psi \rangle \langle\psi | \partial_j \psi \rangle].

                - ``DerivativeType.IMAG`` computes

                .. math::

                    \mathrm{Re(QGT)}_{ij}= \mathrm{Im}[\langle \partial_i \psi | \partial_j \psi \rangle
                        - \langle\partial_i \psi | \psi \rangle \langle\psi | \partial_j \psi \rangle].

                - ``DerivativeType.COMPLEX`` computes

                .. math::

                    \mathrm{QGT}_{ij}= [\langle \partial_i \psi | \partial_j \psi \rangle
                        - \langle\partial_i \psi | \psi \rangle \langle\psi | \partial_j \psi \rangle].
            transpiler: An optional object with a `run` method allowing to transpile the circuits
                that are produced by the internal gradient of this algorithm. If set to `None`,
                these won't be transpiled.
        """
        super().__init__(
            estimator,
            phase_fix,
            derivative_type,
            pass_manager=pass_manager,
        )
        self._gradient = LinCombEstimatorGradient(
            estimator,
            derivative_type=DerivativeType.COMPLEX,
            pass_manager=pass_manager,
        )
        self._lin_comb_qgt_circuit_cache: dict[
            tuple, dict[tuple[Parameter, Parameter], QuantumCircuit]
        ] = {}

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        *,
        precision: float | Sequence[float] | None,
    ) -> QGTResult:
        """Compute the QGT on the given circuits."""
        g_circuits, g_parameter_values, g_parameters = self._preprocess(
            circuits, parameter_values, parameters, self.SUPPORTED_GATES
        )
        results = self._run_unique(
            g_circuits, g_parameter_values, g_parameters, precision=precision
        )
        return self._postprocess(results, circuits, parameter_values, parameters)

    def _run_unique(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        *,
        precision: float | Sequence[float] | None,
    ) -> QGTResult:
        """Compute the QGTs on the given circuits."""
        metadata = []
        all_n, all_m = [], []
        phase_fixes: list[int | np.ndarray] = []

        has_transformed_precision = False

        if isinstance(precision, float) or precision is None:
            precision = [precision] * len(circuits)
            has_transformed_precision = True

        pubs = []

        if not (len(circuits) == len(parameters) == len(parameter_values) == len(precision)):
            raise ValueError(
                f"circuits, parameters, parameter_values and precision must have the same length, but "
                f"have respective lengths {len(circuits)},  {len(parameters)}, {len(parameter_values)} "
                f"and {len(precision)}."
            )

        for circuit, parameter_values_, parameters_, precision_ in zip(
            circuits, parameter_values, parameters, precision
        ):
            # Prepare circuits for the gradient of the specified parameters.
            parameters_ = [p for p in circuit.parameters if p in parameters_]
            meta = {"parameters": parameters_}
            metadata.append(meta)

            # Compute the first term in the QGT
            circuit_key = circuit_cache_key(circuit)
            if circuit_key not in self._lin_comb_qgt_circuit_cache:
                # generate the all of the circuits for the first term in the QGT and cache them.
                # Only the circuit related to specified parameters will be executed.
                # In the future, we can generate the specified circuits on demand.
                self._lin_comb_qgt_circuit_cache[circuit_key] = _make_lin_comb_qgt_circuit(circuit)
            lin_comb_qgt_circuits = self._lin_comb_qgt_circuit_cache[circuit_key]

            qgt_circuits = []
            rows, cols = np.triu_indices(len(parameters_))
            for row, col in zip(rows, cols):
                param_i = parameters_[row]
                param_j = parameters_[col]
                qgt_circuits.append(lin_comb_qgt_circuits[(param_i, param_j)])

            observable = SparsePauliOp.from_list([("I" * circuit.num_qubits, 1)])
            observable_1, observable_2 = _make_lin_comb_observables(
                observable, self._derivative_type
            )

            n = len(qgt_circuits)
            if self._derivative_type == DerivativeType.COMPLEX:
                all_m.append(len(parameters_))
                all_n.append(2 * n)
                pubs.extend(
                    [
                        (qgt_circuit, observable_1, parameter_values_, precision_)
                        for qgt_circuit in qgt_circuits
                    ]
                )
                pubs.extend(
                    [
                        (qgt_circuit, observable_2, parameter_values_, precision_)
                        for qgt_circuit in qgt_circuits
                    ]
                )
            else:
                all_m.append(len(parameters_))
                all_n.append(n)
                pubs.extend(
                    [
                        (qgt_circuit, observable_1, parameter_values_, precision_)
                        for qgt_circuit in qgt_circuits
                    ]
                )

        if self._pass_manager is not None:
            for index, pub in enumerate(pubs):
                new_circuit = self._pass_manager.run(pub[0], **self._pass_manager_options)
                new_observable = pub[1].apply_layout(new_circuit.layout)
                pubs[index] = (new_circuit, new_observable) + pub[2:]

        # Run the single job with all circuits.
        job = self._estimator.run(pubs)

        if self._phase_fix:
            # Compute the second term in the QGT if phase fix is enabled.
            phase_fix_obs = [
                SparsePauliOp.from_list([("I" * circuit.num_qubits, 1)]) for circuit in circuits
            ]
            phase_fix_job = self._gradient.run(
                circuits=circuits,
                observables=phase_fix_obs,
                parameter_values=parameter_values,
                parameters=parameters,
                precision=precision,
            )

        try:
            results = job.result()
            if self._phase_fix:
                gradient_results = phase_fix_job.result()
        except AlgorithmError as exc:
            raise AlgorithmError("Estimator job or gradient job failed.") from exc

        # Compute the phase fix
        if self._phase_fix:
            for gradient in gradient_results.gradients:
                phase_fix = np.outer(np.conjugate(gradient), gradient)
                # Select the real or imaginary part of the phase fix if needed
                if self.derivative_type == DerivativeType.REAL:
                    phase_fix = np.real(phase_fix)
                elif self.derivative_type == DerivativeType.IMAG:
                    phase_fix = np.imag(phase_fix)
                phase_fixes.append(phase_fix)
        else:
            phase_fixes = [0 for _ in range(len(circuits))]
        # Compute the QGT
        qgts = []
        partial_sum_n = 0
        for phase_fix, n, m in zip(phase_fixes, all_n, all_m):  # type: ignore
            qgt = np.zeros((m, m), dtype="complex")
            # Compute the first term in the QGT
            if self.derivative_type == DerivativeType.COMPLEX:
                qgt[np.triu_indices(m)] = np.array(
                    [result.data.evs for result in results[partial_sum_n : partial_sum_n + n // 2]]
                )
                qgt[np.triu_indices(m)] += 1j * np.array(
                    [
                        result.data.evs
                        for result in results[partial_sum_n + n // 2 : partial_sum_n + n]
                    ]
                )
            elif self.derivative_type == DerivativeType.REAL:
                qgt[np.triu_indices(m)] = np.real(
                    [result.data.evs for result in results[partial_sum_n : partial_sum_n + n]]
                )
            elif self.derivative_type == DerivativeType.IMAG:
                qgt[np.triu_indices(m)] = 1j * np.real(
                    [result.data.evs for result in results[partial_sum_n : partial_sum_n + n]]
                )

            # Add the conjugate of the upper triangle to the lower triangle
            qgt += np.triu(qgt, k=1).conjugate().T
            if self.derivative_type == DerivativeType.REAL:
                qgt = np.real(qgt)
            elif self.derivative_type == DerivativeType.IMAG:
                qgt = np.imag(qgt)

            # Subtract the phase fix from the QGT
            qgt -= phase_fix
            partial_sum_n += n
            qgts.append(qgt / 4)

        if has_transformed_precision:
            precision = precision[0]

            if precision is None:
                precision = results[0].metadata["target_precision"]
        else:
            for i, (precision_, result) in enumerate(zip(precision, results)):
                if precision_ is None:
                    precision[i] = results[i].metadata["target_precision"]  # type: ignore

        return QGTResult(
            qgts=qgts, derivative_type=self.derivative_type, metadata=metadata, precision=precision
        )
