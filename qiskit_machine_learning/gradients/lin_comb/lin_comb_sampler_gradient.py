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
Gradient of probabilities with linear combination of unitaries (LCU)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseSamplerV2
from qiskit.providers import Options
from qiskit.result import QuasiDistribution
from qiskit.transpiler.passmanager import BasePassManager

from ...exceptions import AlgorithmError
from ...utils import circuit_cache_key
from ..base.base_sampler_gradient import BaseSamplerGradient
from ..base.sampler_gradient_result import SamplerGradientResult
from ..utils import _make_lin_comb_gradient_circuit


class LinCombSamplerGradient(BaseSamplerGradient):
    """Compute the gradients of the sampling probability.
    This method employs a linear combination of unitaries [1].

    **Reference:**
    [1] Schuld et al., Evaluating analytic gradients on quantum hardware, 2018
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
        sampler: BaseSamplerV2,
        options: Options | None = None,
        pass_manager: BasePassManager | None = None,
    ):
        """
        Args:
            sampler: The sampler used to compute the gradients.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting.
            pass_manager: The pass manager to transpile the circuits if necessary.
                Defaults to ``None``, as some primitives do not need transpiled circuits.
        """
        self._lin_comb_cache: dict[str | tuple, dict[Parameter, QuantumCircuit]] = {}
        super().__init__(sampler, options, pass_manager=pass_manager)

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
    ) -> SamplerGradientResult:  # pragma: no cover
        """Compute the sampler gradients on the given circuits."""
        job_circuits, job_param_values, metadata = [], [], []
        all_n = []
        for circuit, parameter_values_, parameters_ in zip(circuits, parameter_values, parameters):
            # Prepare circuits for the gradient of the specified parameters.
            # TODO: why is this not wrapped into another list level like it is done elsewhere?
            metadata.append({"parameters": parameters_})
            circuit_key = circuit_cache_key(circuit)
            if circuit_key not in self._lin_comb_cache:
                # Cache the circuits for the linear combination of unitaries.
                # We only cache the circuits for the specified parameters in the future.
                self._lin_comb_cache[circuit_key] = _make_lin_comb_gradient_circuit(
                    circuit, add_measurement=True
                )
            lin_comb_circuits = self._lin_comb_cache[circuit_key]
            gradient_circuits = []
            for param_ in parameters_:
                # TODO: the uuid attribute of param_ doesn't match that of param_match
                # TODO: causing the two objects to not be identical, even if all other attrs match
                for param_match in lin_comb_circuits.keys():
                    if param_match.name == param_.name:
                        param = param_match
                gradient_circuits.append(lin_comb_circuits[param])
            # Combine inputs into a single job to reduce overhead.
            n = len(gradient_circuits)
            job_circuits.extend(gradient_circuits)
            job_param_values.extend([parameter_values_] * n)
            all_n.append(n)

        opt = options
        # Run the single job with all circuits.
        if self._pass_manager is None:
            circs = job_circuits
            _len_quasi_dist = 2 ** job_circuits[0].num_qubits
        else:
            circs = self._pass_manager.run(job_circuits)
            _len_quasi_dist = 2 ** circs[0].layout._input_qubit_count
        circ_params = [(circs[i], job_param_values[i]) for i in range(len(job_param_values))]
        job = self._sampler.run(circ_params)
        try:
            results = job.result()
        except Exception as exc:
            raise AlgorithmError("Sampler job failed.") from exc

        # Compute the gradients.
        gradients = []
        partial_sum_n = 0
        for i, n in enumerate(all_n):
            gradient = []
            result = []
            for x in range(partial_sum_n, partial_sum_n + n):
                bitstring_counts = results[x].join_data().get_counts()

                # Normalize the counts to probabilities
                total_shots = sum(bitstring_counts.values())
                probabilities = {k: v / total_shots for k, v in bitstring_counts.items()}

                # Convert to quasi-probabilities
                counts = QuasiDistribution(probabilities)
                result.append({k: v for k, v in counts.items() if int(k) < _len_quasi_dist})
            m = 2 ** circuits[i].num_qubits
            for dist in result:
                grad_dist: dict[int, float] = defaultdict(float)
                for key, value in dist.items():
                    if key < m:
                        grad_dist[key] += value
                    else:
                        grad_dist[key - m] -= value
                gradient.append(dict(grad_dist))
            gradients.append(gradient)
            partial_sum_n += n

        return SamplerGradientResult(gradients=gradients, metadata=metadata, options=opt)
