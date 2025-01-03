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

"""Gradient of Sampler with Finite difference method."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit

from qiskit.primitives import BaseSampler, BaseSamplerV1
from qiskit.primitives.base import BaseSamplerV2
from qiskit.result import QuasiDistribution
from qiskit.providers import Options
from qiskit.transpiler.passmanager import BasePassManager

from ..base.base_sampler_gradient import BaseSamplerGradient
from ..base.sampler_gradient_result import SamplerGradientResult

from ...exceptions import AlgorithmError


class SPSASamplerGradient(BaseSamplerGradient):
    """
    Compute the gradients of the sampling probability by the Simultaneous Perturbation Stochastic
    Approximation (SPSA) [1].

    **Reference:**
    [1] J. C. Spall, Adaptive stochastic approximation by the simultaneous perturbation method in
    IEEE Transactions on Automatic Control, vol. 45, no. 10, pp. 1839-1853, Oct 2020,
    `doi: 10.1109/TAC.2000.880982 <https://ieeexplore.ieee.org/document/880982>`_.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        sampler: BaseSampler,
        epsilon: float = 1e-6,
        batch_size: int = 1,
        seed: int | None = None,
        options: Options | None = None,
        pass_manager: BasePassManager | None = None,
    ):
        """
        Args:
            sampler: The sampler used to compute the gradients.
            epsilon: The offset size for the SPSA gradients.
            batch_size: number of gradients to average.
            seed: The seed for a random perturbation vector.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting
            pass_manager: The pass manager to transpile the circuits if necessary.
                Defaults to ``None``, as some primitives do not need transpiled circuits.

        Raises:
            ValueError: If ``epsilon`` is not positive.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon ({epsilon}) should be positive.")
        self._batch_size = batch_size
        self._epsilon = epsilon
        self._seed = np.random.default_rng(seed)

        super().__init__(sampler, options, pass_manager=pass_manager)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> SamplerGradientResult:  # pragma: no cover
        """Compute the sampler gradients on the given circuits."""
        job_circuits, job_param_values, metadata, offsets = [], [], [], []
        all_n = []
        for circuit, parameter_values_, parameters_ in zip(circuits, parameter_values, parameters):
            # Indices of parameters to be differentiated.
            indices = [circuit.parameters.data.index(p) for p in parameters_]
            metadata.append({"parameters": parameters_})
            offset = np.array(
                [
                    (-1) ** (self._seed.integers(0, 2, len(circuit.parameters)))
                    for _ in range(self._batch_size)
                ]
            )
            plus = [parameter_values_ + self._epsilon * offset_ for offset_ in offset]
            minus = [parameter_values_ - self._epsilon * offset_ for offset_ in offset]
            offsets.append(offset)

            # Combine inputs into a single job to reduce overhead.
            n = 2 * self._batch_size
            job_circuits.extend([circuit] * n)
            job_param_values.extend(plus + minus)
            all_n.append(n)

        opt = options
        # Run the single job with all circuits.
        if isinstance(self._sampler, BaseSamplerV1):
            job = self._sampler.run(job_circuits, job_param_values, **options)
            opt = self._get_local_options(options)
        elif isinstance(self._sampler, BaseSamplerV2):
            if self._pass_manager is None:
                _circs = job_circuits
                _len_quasi_dist = 2 ** job_circuits[0].num_qubits
            else:
                _circs = self._pass_manager.run(job_circuits)
                _len_quasi_dist = 2 ** _circs[0].layout._input_qubit_count
            _circ_params = [(_circs[i], job_param_values[i]) for i in range(len(job_param_values))]
            job = self._sampler.run(_circ_params)
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
        result = []
        partial_sum_n = 0
        for i, n in enumerate(all_n):
            dist_diffs = {}
            if isinstance(self._sampler, BaseSamplerV1):
                result = results.quasi_dists[partial_sum_n : partial_sum_n + n]
            elif isinstance(self._sampler, BaseSamplerV2):
                _result = []
                for m in range(partial_sum_n, partial_sum_n + n):
                    if hasattr(results[i].data, "meas"):
                        _bitstring_counts = results[m].data.meas.get_counts()
                    else:
                        # Fallback to 'c' if 'meas' is not available.
                        _bitstring_counts = results[m].data.c.get_counts()
                    # Normalize the counts to probabilities
                    _total_shots = sum(_bitstring_counts.values())
                    _probabilities = {k: v / _total_shots for k, v in _bitstring_counts.items()}
                    # Convert to quasi-probabilities
                    _counts = QuasiDistribution(_probabilities)
                    _result.append({k: v for k, v in _counts.items() if int(k) < _len_quasi_dist})
                    result = [{key: d[key] for key in sorted(d)} for d in _result]

            for j, (dist_plus, dist_minus) in enumerate(zip(result[: n // 2], result[n // 2 :])):
                dist_diff: dict[int, float] = defaultdict(float)
                for key, value in dist_plus.items():
                    dist_diff[key] += value / (2 * self._epsilon)
                for key, value in dist_minus.items():
                    dist_diff[key] -= value / (2 * self._epsilon)
                dist_diffs[j] = dist_diff

            gradient = []
            indices = [circuits[i].parameters.data.index(p) for p in metadata[i]["parameters"]]
            for j in indices:
                gradient_j: dict[int, float] = defaultdict(float)
                for k in range(self._batch_size):
                    for key, value in dist_diffs[k].items():
                        gradient_j[key] += value * offsets[i][k][j]
                gradient_j = {key: value / self._batch_size for key, value in gradient_j.items()}
                gradient.append(gradient_j)
            gradients.append(gradient)
            partial_sum_n += n

        return SamplerGradientResult(gradients=gradients, metadata=metadata, options=opt)
