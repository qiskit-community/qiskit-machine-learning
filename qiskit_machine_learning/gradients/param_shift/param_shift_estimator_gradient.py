# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Gradient of probabilities with parameter shift
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.primitives import BaseEstimatorV1
from qiskit.providers.options import Options

from ..base.base_estimator_gradient import BaseEstimatorGradient
from ..base.estimator_gradient_result import EstimatorGradientResult
from ..utils import _make_param_shift_parameter_values
from ...exceptions import QiskitMachineLearningError


class ParamShiftEstimatorGradient(BaseEstimatorGradient):
    """
    Compute the gradients of the expectation values by the parameter shift rule [1].

    **Reference:**
    [1] Schuld, M., Bergholm, V., Gogolin, C., Izaac, J., and Killoran, N. Evaluating analytic
    gradients on quantum hardware, `DOI <https://doi.org/10.1103/PhysRevA.99.032331>`_
    """

    SUPPORTED_GATES = [
        "x",
        "y",
        "z",
        "h",
        "rx",
        "ry",
        "rz",
        "p",
        "cx",
        "cy",
        "cz",
        "ryy",
        "rxx",
        "rzz",
        "rzx",
    ]

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]] | np.ndarray,
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the gradients of the expectation values by the parameter shift rule."""
        g_circuits, g_parameter_values, g_parameters = self._preprocess(
            circuits, parameter_values, parameters, self.SUPPORTED_GATES
        )
        results = self._run_unique(
            g_circuits, observables, g_parameter_values, g_parameters, **options
        )
        return self._postprocess(results, circuits, parameter_values, parameters)

    def _run_unique(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        job_circuits, job_observables, job_param_values, metadata = [], [], [], []
        all_n = []
        for circuit, observable, parameter_values_, parameters_ in zip(
            circuits, observables, parameter_values, parameters
        ):
            metadata.append({"parameters": parameters_})
            # Make parameter values for the parameter shift rule.
            param_shift_parameter_values = _make_param_shift_parameter_values(
                circuit, parameter_values_, parameters_
            )
            # Combine inputs into a single job to reduce overhead.
            n = len(param_shift_parameter_values)
            job_circuits.extend([circuit] * n)
            job_observables.extend([observable] * n)
            job_param_values.extend(param_shift_parameter_values)
            all_n.append(n)

        # Determine how to run the estimator based on its version
        if isinstance(self._estimator, BaseEstimatorV1):
            # Run the single job with all circuits.
            job = self._estimator.run(
                job_circuits,
                job_observables,
                job_param_values,
                **options,
            )
            results = job.result()

            # Compute the gradients.
            gradients = []
            partial_sum_n = 0
            for n in all_n:
                result = results.values[partial_sum_n : partial_sum_n + n]
                gradient_ = (result[: n // 2] - result[n // 2 :]) / 2
                gradients.append(gradient_)
                partial_sum_n += n

            opt = self._get_local_options(options)

        elif isinstance(self._estimator, BaseEstimatorV2):
            if self._pass_manager is None:
                circs = job_circuits
                observables = job_observables
            else:
                circs = self._pass_manager.run(job_circuits)
                observables = [
                    op.apply_layout(circs[i].layout) for i, op in enumerate(job_observables)
                ]
            # Prepare circuit-observable-parameter tuples (PUBs)
            circuit_observable_params = []
            for pub in zip(circs, observables, job_param_values):
                circuit_observable_params.append(pub)

            # For BaseEstimatorV2, run the estimator using PUBs and specified precision
            job = self._estimator.run(circuit_observable_params)
            results = job.result()
            results = np.array([float(r.data.evs) for r in results])

            # Compute the gradients.
            gradients = []
            partial_sum_n = 0
            for n in all_n:
                result = results[partial_sum_n : partial_sum_n + n]
                gradient_ = (result[: n // 2] - result[n // 2 :]) / 2
                gradients.append(gradient_)
                partial_sum_n += n

            opt = Options(**options)

        else:
            raise QiskitMachineLearningError(
                "The accepted estimators are BaseEstimatorV1 and BaseEstimatorV2; got "
                + f"{type(self._estimator)} instead. Note that BaseEstimatorV1 is deprecated in"
                + "Qiskit and removed in Qiskit IBM Runtime."
            )

        return EstimatorGradientResult(gradients=gradients, metadata=metadata, options=opt)
