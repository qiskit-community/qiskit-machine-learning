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
Compute-uncompute fidelity interface using primitives
"""

from __future__ import annotations
from collections.abc import Sequence
from copy import copy

from qiskit import QuantumCircuit
from qiskit.primitives import BaseSampler, BaseSamplerV1, SamplerResult, StatevectorSampler
from qiskit.primitives.base import BaseSamplerV2
from qiskit.transpiler.passmanager import PassManager
from qiskit.result import QuasiDistribution
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.providers import Options

from ..exceptions import AlgorithmError, QiskitMachineLearningError
from ..utils.deprecation import issue_deprecation_msg
from .base_state_fidelity import BaseStateFidelity
from .state_fidelity_result import StateFidelityResult
from ..algorithm_job import AlgorithmJob


class ComputeUncompute(BaseStateFidelity):
    r"""
    This class leverages the sampler primitive to calculate the state
    fidelity of two quantum circuits following the compute-uncompute
    method (see [1] for further reference).
    The fidelity can be defined as the state overlap.

    .. math::

            |\langle\psi(x)|\phi(y)\rangle|^2

    where :math:`x` and :math:`y` are optional parametrizations of the
    states :math:`\psi` and :math:`\phi` prepared by the circuits
    ``circuit_1`` and ``circuit_2``, respectively.

    **Reference:**
    [1] Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala,
    A., Chow, J. M., & Gambetta, J. M. (2019). Supervised learning
    with quantum-enhanced feature spaces. Nature, 567(7747), 209-212.
    `arXiv:1804.11326v2 [quant-ph] <https://arxiv.org/pdf/1804.11326.pdf>`_

    """

    def __init__(
        self,
        sampler: BaseSampler | BaseSamplerV2,
        *,
        num_virtual_qubits: int | None = None,
        pass_manager: PassManager | None = None,
        options: Options | None = None,
        local: bool = False,
    ) -> None:
        r"""
        Args:
            sampler: Sampler primitive instance.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > fidelity's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting.
            local: If set to ``True``, the fidelity is averaged over
                single-qubit projectors

                .. math::

                    \hat{O} = \frac{1}{N}\sum_{i=1}^N|0_i\rangle\langle 0_i|,

                instead of the global projector :math:`|0\rangle\langle 0|^{\otimes n}`.
                This coincides with the standard (global) fidelity in the limit of
                the fidelity approaching 1. Might be used to increase the variance
                to improve trainability in algorithms such as :class:`~.time_evolvers.PVQD`.

        Raises:
            ValueError: If the sampler is not an instance of ``BaseSampler``.
        """
        if (not isinstance(sampler, BaseSampler)) and (not isinstance(sampler, BaseSamplerV2)):
            raise ValueError(
                f"The sampler should be an instance of BaseSampler or BaseSamplerV2, "
                f"but got {type(sampler)}"
            )
        if (
            isinstance(sampler, BaseSamplerV2)
            and (pass_manager is None)
            and not isinstance(sampler, StatevectorSampler)
        ):
            raise ValueError(f"A pass_manager should be provided for {type(sampler)}.")
        if (pass_manager is not None) and (num_virtual_qubits is None):
            raise ValueError(
                f"Number of virtual qubits should be provided for {type(pass_manager)}."
            )
        if isinstance(sampler, BaseSamplerV1):
            issue_deprecation_msg(
                msg="V1 Primitives are deprecated",
                version="0.8.0",
                remedy="Use V2 primitives for continued compatibility and support.",
                period="4 months",
            )
        self._sampler: BaseSampler = sampler
        self.num_virtual_qubits = num_virtual_qubits
        self.pass_manager = pass_manager
        self._local = local
        self._default_options = Options()
        if options is not None:
            self._default_options.update_options(**options)
        super().__init__()

    def create_fidelity_circuit(
        self, circuit_1: QuantumCircuit, circuit_2: QuantumCircuit
    ) -> QuantumCircuit:
        """
        Combines ``circuit_1`` and ``circuit_2`` to create the
        fidelity circuit following the compute-uncompute method.

        Args:
            circuit_1: (Parametrized) quantum circuit.
            circuit_2: (Parametrized) quantum circuit.

        Returns:
            The fidelity quantum circuit corresponding to circuit_1 and circuit_2.
        """
        if len(circuit_1.clbits) > 0:
            circuit_1.remove_final_measurements()
        if len(circuit_2.clbits) > 0:
            circuit_2.remove_final_measurements()

        circuit = circuit_1.compose(circuit_2.inverse())
        circuit.measure_all()
        if self.pass_manager is not None:
            circuit = self.pass_manager.run(circuit)
        return circuit

    def _run(
        self,
        circuits_1: QuantumCircuit | Sequence[QuantumCircuit],
        circuits_2: QuantumCircuit | Sequence[QuantumCircuit],
        values_1: Sequence[float] | Sequence[Sequence[float]] | None = None,
        values_2: Sequence[float] | Sequence[Sequence[float]] | None = None,
        **options,
    ) -> AlgorithmJob:
        r"""
        Computes the state overlap (fidelity) calculation between two
        (parametrized) circuits (first and second) for a specific set of parameter
        values (first and second) following the compute-uncompute method.

        Args:
            circuits_1: (Parametrized) quantum circuits preparing :math:`|\psi\rangle`.
            circuits_2: (Parametrized) quantum circuits preparing :math:`|\phi\rangle`.
            values_1: Numerical parameters to be bound to the first circuits.
            values_2: Numerical parameters to be bound to the second circuits.
            options: Primitive backend runtime options used for circuit execution.
                    The order of priority is: options in ``run`` method > fidelity's
                    default options > primitive's default setting.
                    Higher priority setting overrides lower priority setting.

        Returns:
            An AlgorithmJob for the fidelity calculation.

        Raises:
            ValueError: At least one pair of circuits must be defined.
            AlgorithmError: If the sampler job is not completed successfully.
        """

        circuits = self._construct_circuits(circuits_1, circuits_2)
        if len(circuits) == 0:
            raise ValueError(
                "At least one pair of circuits must be defined to calculate the state overlap."
            )
        values = self._construct_value_list(circuits_1, circuits_2, values_1, values_2)

        # The priority of run options is as follows:
        # options in `evaluate` method > fidelity's default options >
        # primitive's default options.
        opts = copy(self._default_options)
        opts.update_options(**options)

        if isinstance(self._sampler, BaseSamplerV1):
            sampler_job = self._sampler.run(
                circuits=circuits, parameter_values=values, **opts.__dict__
            )
            local_opts = self._get_local_options(opts.__dict__)
        elif isinstance(self._sampler, BaseSamplerV2):
            sampler_job = self._sampler.run(
                [(circuits[i], values[i]) for i in range(len(circuits))], **opts.__dict__
            )
            local_opts = opts.__dict__
        else:
            raise QiskitMachineLearningError(
                "The accepted estimators are BaseSamplerV1 (deprecated) and BaseSamplerV2; got"
                + f" {type(self._sampler)} instead."
            )
        return AlgorithmJob(
            ComputeUncompute._call,
            sampler_job,
            circuits,
            self._local,
            local_opts,
            self._sampler,
            self._post_process_v2,
            self.num_virtual_qubits,
        )

    @staticmethod
    def _call(
        job: PrimitiveJob,
        circuits: Sequence[QuantumCircuit],
        local: bool,
        local_opts: Options = None,
        _sampler=None,
        _post_process_v2=None,
        num_virtual_qubits=None,
    ) -> StateFidelityResult:
        try:
            result = job.result()
        except Exception as exc:
            raise AlgorithmError("Sampler job failed!") from exc

        if isinstance(_sampler, BaseSamplerV1):
            quasi_dists = result.quasi_dists
        elif isinstance(_sampler, BaseSamplerV2):
            quasi_dists = _post_process_v2(result)

        if local:
            raw_fidelities = [
                ComputeUncompute._get_local_fidelity(
                    prob_dist,
                    (
                        num_virtual_qubits
                        if isinstance(_sampler, BaseSamplerV2)
                        else circuit.num_qubits
                    ),
                )
                for prob_dist, circuit in zip(quasi_dists, circuits)
            ]
        else:
            raw_fidelities = [
                ComputeUncompute._get_global_fidelity(prob_dist) for prob_dist in quasi_dists
            ]
        fidelities = ComputeUncompute._truncate_fidelities(raw_fidelities)

        return StateFidelityResult(
            fidelities=fidelities,
            raw_fidelities=raw_fidelities,
            metadata=result.metadata,
            options=local_opts,
        )

    @property
    def options(self) -> Options:
        """Return the union of estimator options setting and fidelity default options,
        where, if the same field is set in both, the fidelity's default options override
        the primitive's default setting.

        Returns:
            The fidelity default + estimator options.
        """
        return self._get_local_options(self._default_options.__dict__)

    def update_default_options(self, **options):
        """Update the fidelity's default options setting.

        Args:
            **options: The fields to update the default options.
        """

        self._default_options.update_options(**options)

    def _get_local_options(self, options: Options) -> Options:
        """Return the union of the primitive's default setting,
        the fidelity default options, and the options in the ``run`` method.
        The order of priority is: options in ``run`` method > fidelity's
                default options > primitive's default setting.

        Args:
            options: The fields to update the options

        Returns:
            The fidelity default + estimator + run options.
        """
        opts = copy(self._sampler.options)
        opts.update_options(**options)
        return opts

    def _post_process_v2(self, result: SamplerResult):
        quasis = []
        for i in range(len(result)):
            bitstring_counts = result[i].data.meas.get_counts()

            # Normalize the counts to probabilities
            total_shots = sum(bitstring_counts.values())
            probabilities = {k: v / total_shots for k, v in bitstring_counts.items()}

            # Convert to quasi-probabilities
            counts = QuasiDistribution(probabilities)
            quasi_probs = {k: v for k, v in counts.items() if int(k) < 2**self.num_virtual_qubits}
            quasis.append(quasi_probs)
        return quasis

    @staticmethod
    def _get_global_fidelity(probability_distribution: dict[int, float]) -> float:
        """Process the probability distribution of a measurement to determine the
        global fidelity.

        Args:
            probability_distribution: Obtained from the measurement result

        Returns:
            The global fidelity.
        """
        return probability_distribution.get(0, 0)

    @staticmethod
    def _get_local_fidelity(probability_distribution: dict[int, float], num_qubits: int) -> float:
        """Process the probability distribution of a measurement to determine the
        local fidelity by averaging over single-qubit projectors.

        Args:
            probability_distribution: Obtained from the measurement result

        Returns:
            The local fidelity.
        """
        fidelity = 0.0
        for qubit in range(num_qubits):
            for bitstring, prob in probability_distribution.items():
                # Check whether the bit representing the current qubit is 0
                if not bitstring >> qubit & 1:
                    fidelity += prob / num_qubits
        return fidelity
