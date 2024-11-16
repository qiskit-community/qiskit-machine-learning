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

from collections import defaultdict
from collections.abc import Sequence

from qiskit.circuit import Parameter, QuantumCircuit

from qiskit.primitives import BaseSamplerV1
from qiskit.primitives.base import BaseSamplerV2
from qiskit.result import QuasiDistribution

from ..base.base_sampler_gradient import BaseSamplerGradient
from ..base.sampler_gradient_result import SamplerGradientResult
from ..utils import _make_param_shift_parameter_values
from ...exceptions import AlgorithmError, QiskitMachineLearningError


class ParamShiftSamplerGradient(BaseSamplerGradient):
    """
    Compute the gradients of the sampling probability by the parameter shift rule [1].

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
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> SamplerGradientResult:
        """Compute the estimator gradients on the given circuits."""
        g_circuits, g_parameter_values, g_parameters = self._preprocess(
            circuits, parameter_values, parameters, self.SUPPORTED_GATES
        )
        results = self._run_unique(g_circuits, g_parameter_values, g_parameters, **options)
        return self._postprocess(results, circuits, parameter_values, parameters)

    def _run_unique(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> SamplerGradientResult:
        """Compute the sampler gradients on the given circuits."""
        job_circuits, job_param_values, metadata = [], [], []
        all_n = []
        for circuit, parameter_values_, parameters_ in zip(circuits, parameter_values, parameters):
            metadata.append({"parameters": parameters_})
            # Make parameter values for the parameter shift rule.
            param_shift_parameter_values = _make_param_shift_parameter_values(
                circuit, parameter_values_, parameters_
            )
            # Combine inputs into a single job to reduce overhead.
            n = len(param_shift_parameter_values)
            job_circuits.extend([circuit] * n)
            job_param_values.extend(param_shift_parameter_values)
            all_n.append(n)

        # Run the single job with all circuits.
        if isinstance(self._sampler, BaseSamplerV1):
            job = self._sampler.run(job_circuits, job_param_values, **options)
        elif isinstance(self._sampler, BaseSamplerV2):
            if self._pass_manager is None:
                circs = job_circuits
                _len_quasi_dist = 2**job_circuits[0].num_qubits
            else:
                circs = self._pass_manager.run(job_circuits)
                _len_quasi_dist = 2**circs[0].layout._input_qubit_count
            circ_params = [
                (circs[i], job_param_values[i]) for i in range(len(job_param_values))
            ]
            job = self._sampler.run(circ_params)
        else:
            raise AlgorithmError(
                "The accepted estimators are BaseSamplerV1 (deprecated) and BaseSamplerV2; got "
                + f"{type(self._sampler)} instead."
            )

        try:
            results = job.result()
        except Exception as exc:
            raise AlgorithmError("Sampler job failed.") from exc

        # Compute the gradients.
        gradients = []
        partial_sum_n = 0
        opt = None  # Required by PyLint: possibly-used-before-assignment
        for n in all_n:
            gradient = []

            if isinstance(self._sampler, BaseSamplerV1):
                result = results.quasi_dists[partial_sum_n : partial_sum_n + n]
                opt = self._get_local_options(options)

            elif isinstance(self._sampler, BaseSamplerV2):
                result = []
                for i in range(partial_sum_n, partial_sum_n + n):
                    bitstring_counts = results[i].data.meas.get_counts()

                    # Normalize the counts to probabilities
                    total_shots = sum(bitstring_counts.values())
                    probabilities = {k: v / total_shots for k, v in bitstring_counts.items()}

                    # Convert to quasi-probabilities
                    counts = QuasiDistribution(probabilities)
                    result.append(
                        {k: v for k, v in counts.items() if int(k) < _len_quasi_dist}
                    )
                    opt = options

            for dist_plus, dist_minus in zip(result[: n // 2], result[n // 2 :]):
                grad_dist: dict[int, float] = defaultdict(float)
                for key, val in dist_plus.items():
                    grad_dist[key] += val / 2
                for key, val in dist_minus.items():
                    grad_dist[key] -= val / 2
                gradient.append(dict(grad_dist))
            gradients.append(gradient)
            partial_sum_n += n

        return SamplerGradientResult(gradients=gradients, metadata=metadata, options=opt)
