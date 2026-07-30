# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Helpers for reproducible local IBM Runtime primitive tests."""

from __future__ import annotations

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2, SamplerV2, Session
from qiskit_ibm_runtime.options import EstimatorOptions, SamplerOptions, SimulatorOptions

DEFAULT_RUNTIME_SEED = 123


def make_generic_backend(num_qubits: int, *, seed: int = DEFAULT_RUNTIME_SEED) -> GenericBackendV2:
    """Return a noise-free fake backend with a fixed layout seed."""
    return GenericBackendV2(num_qubits=num_qubits, noise_info=False, seed=seed)


def make_simulator_options(*, seed: int = DEFAULT_RUNTIME_SEED) -> SimulatorOptions:
    """Return simulator options for deterministic local Runtime execution."""
    return SimulatorOptions(seed_simulator=seed)


def make_estimator_options(
    *, seed: int = DEFAULT_RUNTIME_SEED, default_shots: int | None = None
) -> EstimatorOptions:
    """Return EstimatorV2 options with deterministic simulation settings."""
    options = EstimatorOptions(
        simulator=make_simulator_options(seed=seed),
        seed_estimator=seed,
        resilience_level=0,
    )
    if default_shots is not None:
        options.default_shots = default_shots
    return options


def make_sampler_options(
    *, seed: int = DEFAULT_RUNTIME_SEED, default_shots: int | None = None
) -> SamplerOptions:
    """Return SamplerV2 options with deterministic simulation settings."""
    options = SamplerOptions(simulator=make_simulator_options(seed=seed))
    if default_shots is not None:
        options.default_shots = default_shots
    return options


def make_runtime_session(
    num_qubits: int, *, seed: int = DEFAULT_RUNTIME_SEED
) -> tuple[GenericBackendV2, Session]:
    """Create a backend and session for local Runtime primitive tests."""
    backend = make_generic_backend(num_qubits, seed=seed)
    return backend, Session(backend=backend)


def make_runtime_pass_manager(
    backend: GenericBackendV2,
    *,
    optimization_level: int = 0,
    seed: int = DEFAULT_RUNTIME_SEED,
    **kwargs,
):
    """Create a pass manager with a fixed transpiler seed."""
    return generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        seed_transpiler=seed,
        **kwargs,
    )


def make_estimator_v2(
    num_qubits: int,
    *,
    seed: int = DEFAULT_RUNTIME_SEED,
    default_shots: int | None = None,
) -> tuple[GenericBackendV2, EstimatorV2]:
    """Create a deterministic EstimatorV2 backed by a local fake backend."""
    backend, session = make_runtime_session(num_qubits, seed=seed)
    estimator = EstimatorV2(
        mode=session,
        options=make_estimator_options(seed=seed, default_shots=default_shots),
    )
    return backend, estimator


def make_sampler_v2(
    num_qubits: int,
    *,
    seed: int = DEFAULT_RUNTIME_SEED,
    default_shots: int | None = None,
) -> tuple[GenericBackendV2, Session, SamplerV2]:
    """Create a deterministic SamplerV2 backed by a local fake backend."""
    backend, session = make_runtime_session(num_qubits, seed=seed)
    sampler = SamplerV2(
        mode=session,
        options=make_sampler_options(seed=seed, default_shots=default_shots),
    )
    return backend, session, sampler
