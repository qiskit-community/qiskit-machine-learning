# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Qiskit Machine Learning Estimator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

from qiskit.primitives.base import BaseEstimatorV2
from qiskit.primitives.containers import DataBin, EstimatorPubLike, PrimitiveResult, PubResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.primitive_job import PrimitiveJob


class QMLEstimator(BaseEstimatorV2):
    """
    EstimatorV2-compatible primitive with V1 Estimator semantics:

      - shots=None (default): exact expectation values from statevector; stds=0.
      - shots=int           : sample each expectation value from N(mean, stderr),
                              where stderr is a normal-approximation standard error.

    Additionally supports the V2 `precision` argument:
      - if shots is None and effective precision > 0: sample from N(mean, precision)
        (similar to other V2 estimator implementations that treat precision as Gaussian noise).
    """

    def __init__(
        self,
        *,
        shots: int | None = None,
        default_precision: float = 0.0,
        seed: np.random.Generator | int | None = None,
        **_kwargs: Any,
    ) -> None:
        self._shots = None if shots is None else int(shots)
        self._default_precision = float(default_precision)
        self._seed = seed

        # Provide a mutable .options like V1 estimators expose (useful for ML glue code).
        self.options = _OptionsNS(
            shots=self._shots,
            default_precision=self._default_precision,
            seed=self._seed,
        )

    @property
    def default_precision(self) -> float:
        return self._default_precision

    @property
    def seed(self) -> np.random.Generator | int | None:
        return self._seed

    @property
    def shots(self) -> int | None:
        return self._shots

    def set_options(self, **fields: Any) -> None:
        """V1-style helper to update options in-place."""
        self.options.update(**fields)
        self._shots = None if getattr(self.options, "shots", None) is None else int(self.options.shots)
        self._default_precision = float(getattr(self.options, "default_precision", self._default_precision))
        self._seed = getattr(self.options, "seed", self._seed)

    def run(
        self,
        pubs: Iterable[EstimatorPubLike],
        *,
        precision: float | None = None,
        shots: int | None = None,
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        """
        Run on PUBs and return V2 PrimitiveResult[PubResult].

        Note: `shots` here is an extra convenience override (like your QMLSampler).
        The official V2 knob is `precision`.
        """
        if precision is None:
            precision = self._default_precision
        eff_shots = self._shots if shots is None else (None if shots is None else int(shots))

        coerced = [EstimatorPub.coerce(pub, float(precision)) for pub in pubs]
        job = PrimitiveJob(self._run, coerced, float(precision), eff_shots)
        job._submit()  # pylint: disable=protected-access
        return job

    # -------------------- implementation --------------------

    def _run(
        self,
        pubs: list[EstimatorPub],
        precision: float,
        shots: int | None,
    ) -> PrimitiveResult[PubResult]:
        return PrimitiveResult(
            [self._run_pub(pub, precision=precision, shots=shots) for pub in pubs],
            metadata={"version": 2},
        )

    def _rng(self) -> np.random.Generator:
        # V1 semantics: if seed is an int, use it to seed a fresh generator (repeatable runs).
        # If it's already a Generator, use it as-is (stateful).
        if isinstance(self._seed, np.random.Generator):
            return self._seed
        return np.random.default_rng(self._seed)

    def _run_pub(self, pub: EstimatorPub, *, precision: float, shots: int | None) -> PubResult:
        circuit = pub.circuit
        observables = pub.observables
        parameter_values = pub.parameter_values

        bound_circuits = parameter_values.bind_all(circuit)
        bc_circuits, bc_obs = np.broadcast_arrays(bound_circuits, observables)

        evs = np.empty(bc_circuits.shape, dtype=np.float64)
        stds = np.zeros(bc_circuits.shape, dtype=np.float64)

        rng = self._rng()

        for idx in np.ndindex(bc_circuits.shape):
            sv = Statevector.from_instruction(bc_circuits[idx])

            # Exact mean expectation value
            obs = _coerce_observable(bc_obs[idx])
            mean = sv.expectation_value(obs)
            mean = float(np.real_if_close(mean))
            ev = mean

            if shots is not None:
                # Normal approximation standard error.
                # For Pauli sums, approximate variance as sum_i |c_i|^2 (1 - <P_i>^2),
                # assuming independent estimation of each term.
                stderr = _stderr_from_sparse_pauli_sum(sv, bc_obs[idx], shots)
                stds[idx] = stderr
                if stderr > 0:
                    ev = float(rng.normal(loc=mean, scale=stderr))
            elif precision and precision > 0.0:
                # Precision-based Gaussian noise mode (optional compatibility)
                stds[idx] = float(precision)
                ev = float(rng.normal(loc=mean, scale=float(precision)))

            evs[idx] = ev

        data = DataBin(evs=evs, stds=stds, shape=evs.shape)
        meta = {
            "shots": shots,
            "target_precision": float(precision),
            "circuit_metadata": getattr(pub, "metadata", {}),
            "exact": shots is None and (precision == 0.0),
        }
        return PubResult(data=data, metadata=meta)



def _coerce_observable(obs: Any) -> Any:
    """Accept extra legacy-ish observable formats (like {'Z': 1.0}) and convert
    them to something Statevector.expectation_value understands."""
    if isinstance(obs, (SparsePauliOp, Operator)):
        return obs
    # qiskit-opflow style (if it still shows up somewhere): PauliSumOp.primitive -> SparsePauliOp
    prim = getattr(obs, "primitive", None)
    if isinstance(prim, SparsePauliOp):
        return prim

    if isinstance(obs, Mapping):
        if not obs:
            raise ValueError("Observable dict is empty.")
        return SparsePauliOp.from_list([(str(label), complex(coeff)) for label, coeff in obs.items()])

    # Common convenience: a Pauli label string, e.g. "Z", "ZI", "XX"
    if isinstance(obs, str):
        return SparsePauliOp.from_list([(obs, 1.0)])

    # Lists/tuples of (label, coeff)
    if isinstance(obs, (list, tuple)) and obs and isinstance(obs[0], (list, tuple)) and len(obs[0]) == 2:
        return SparsePauliOp.from_list([(str(lbl), complex(c)) for (lbl, c) in obs])

    # Try SparsePauliOp constructor for Pauli/PauliList/etc
    try:
        return SparsePauliOp(obs)
    except Exception:
        pass

    # Fallback to general operator conversion (matrices, etc.)
    return Operator(obs)


def _stderr_from_sparse_pauli_sum(sv: Statevector, obs: Any, shots: int) -> float:
    spo = _coerce_observable(obs)
    if not isinstance(spo, SparsePauliOp) or shots <= 0:
        return 0.0

    ev_terms = np.array([float(np.real_if_close(sv.expectation_value(p))) for p in spo.paulis], dtype=float)
    coeffs = np.asarray(spo.coeffs)
    var = np.sum(np.abs(coeffs) ** 2 * np.maximum(0.0, 1.0 - ev_terms**2))
    return float(np.sqrt(var / float(shots)))


def _options_to_dict(opts) -> dict:
    if opts is None:
        return {}
    if is_dataclass(opts):
        return asdict(opts)  # type: ignore
    if hasattr(opts, "__dict__"):
        return {k: v for k, v in vars(opts).items() if not k.startswith("_")}
    return {}


class _OptionsNS(SimpleNamespace):
    """Mutable options name space with update(**kwargs)."""

    def update(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
