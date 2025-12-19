# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files must carry a notice indicating
# that they have been altered from the originals.
"""Qiskit Machine Learning estimator primitive.

This module provides a small wrapper around Qiskit's ``StatevectorEstimator``
that offers switch between:

* Exact mode (``default_precision == 0``): analytic expectation values with
  deterministic outputs and zero standard deviation.
* Delegate mode (``default_precision != 0``): defer execution to
  ``StatevectorEstimator`` (precision-aware reference implementation).

"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives.containers import (
    DataBin,
    EstimatorPubLike,
    PrimitiveResult,
    PubResult,
)
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector


class QMLEstimator(StatevectorEstimator):
    """Statevector-based estimator with two execution modes.

    Modes are selected at construction time:

    * ``default_precision == 0.0`` (default): exact mode: Results are
      deterministic (analytic expectation values) with ``stds == 0``.
      Any per-call ``precision`` override is accepted for API compatibility
      but ignored.
    * ``default_precision != 0.0``: delegate mode: Execution is delegated
      to :class:`~qiskit.primitives.StatevectorEstimator`, which interprets the
      precision parameter according to the reference primitive behavior.
    """

    def __init__(
        self,
        *,
        default_precision: float = 0.0,
        seed: np.random.Generator | int | None = None,
        **kwargs: Any,
    ) -> None:
        if float(default_precision) == 0.0:
            self._exact_mode = True
        else:
            self._exact_mode = False

        if self._exact_mode:
            super().__init__(default_precision=0.0, seed=seed, **kwargs)
        else:
            super().__init__(default_precision=float(default_precision), seed=seed, **kwargs)

        # Provide a mutable, V1-style `.options` namespace for ML integrations.
        parent_opts = object.__getattribute__(self, "__dict__").get("options", None)
        base = _options_to_dict(parent_opts)
        merged = dict(base)
        merged.setdefault("default_precision", float(default_precision))
        merged.setdefault("seed", seed)
        self.options = _OptionsNS(**merged)

    def run(
        self,
        pubs: Iterable[EstimatorPubLike],
        *,
        precision: float | None = None,
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        """Evaluate a collection of estimator PUBs.

        Args:
            pubs: Iterable of PUB-like inputs describing circuits, observables,
                and parameter values.
            precision: Target precision for V2-style estimation. In exact mode,
                this value is ignored and results are deterministic.

        Returns:
            A job that yields a ``PrimitiveResult[PubResult]``.
        """
        if not self._exact_mode:
            return super().run(pubs, precision=precision)

        coerced = [EstimatorPub.coerce(pub, 0.0) for pub in pubs]  # satisfy validation
        job: PrimitiveJob[PrimitiveResult[PubResult]] = PrimitiveJob(self._run_exact, coerced)
        job._submit()  # pylint: disable=protected-access
        return job

    # -------------------- exact-mode implementation --------------------

    def _run_exact(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        return PrimitiveResult(
            [self._run_pub_exact(pub) for pub in pubs],
            metadata={"version": 2},
        )

    def _run_pub_exact(self, pub: EstimatorPub) -> PubResult:
        circuit = pub.circuit
        observables = pub.observables
        parameter_values = pub.parameter_values

        bound_circuits = parameter_values.bind_all(circuit)
        bc_circuits, bc_obs = np.broadcast_arrays(bound_circuits, observables)

        evs = np.empty(bc_circuits.shape, dtype=np.float64)
        stds = np.zeros(bc_circuits.shape, dtype=np.float64)

        for idx in np.ndindex(bc_circuits.shape):
            sv = Statevector.from_instruction(bc_circuits[idx])
            obs = _coerce_observable(bc_obs[idx])
            mean = sv.expectation_value(obs)
            evs[idx] = float(np.real_if_close(mean))

        data = DataBin(evs=evs, stds=stds, shape=evs.shape)
        meta = {
            "shots": None,
            "target_precision": 0.0,
            "circuit_metadata": getattr(pub, "metadata", {}),
            "exact": True,
        }
        return PubResult(data=data, metadata=meta)


def _coerce_observable(obs: Any) -> Any:
    """Normalize supported observable formats.

    Converts common encodings into objects accepted by
    :meth:`qiskit.quantum_info.Statevector.expectation_value`.
    """
    if isinstance(obs, (SparsePauliOp, Operator)):
        return obs

    prim = getattr(obs, "primitive", None)
    if isinstance(prim, SparsePauliOp):
        return prim

    if isinstance(obs, Mapping):
        if not obs:
            raise ValueError("Observable mapping is empty.")
        return SparsePauliOp.from_list([(str(label), complex(coeff)) for label, coeff in obs.items()])

    if isinstance(obs, str):
        return SparsePauliOp.from_list([(obs, 1.0)])

    if isinstance(obs, (list, tuple)) and obs and isinstance(obs[0], (list, tuple)) and len(obs[0]) == 2:
        return SparsePauliOp.from_list([(str(lbl), complex(c)) for (lbl, c) in obs])

    try:
        return SparsePauliOp(obs)
    except Exception:
        return Operator(obs)


def _options_to_dict(opts: Any) -> dict[str, Any]:
    """Best-effort conversion of an options-like object into a plain dict."""
    if opts is None:
        return {}
    if is_dataclass(opts):
        return dict(asdict(opts))
    if isinstance(opts, Mapping):
        return dict(opts)
    to_dict = getattr(opts, "to_dict", None)
    if callable(to_dict):
        try:
            return dict(to_dict())
        except Exception:
            pass
    try:
        return dict(vars(opts))
    except TypeError:
        return {}


class _OptionsNS(SimpleNamespace):
    """Mutable options namespace supporting ``update(**kwargs)``."""

    def update(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
