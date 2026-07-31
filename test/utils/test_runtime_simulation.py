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
"""Tests for runtime simulation helpers."""

from unittest.mock import patch

from test import QiskitMachineLearningTestCase
from test.utils.runtime_simulation import (
    DEFAULT_RUNTIME_SEED,
    make_estimator_options,
    make_estimator_v2,
    make_generic_backend,
    make_runtime_pass_manager,
    make_runtime_session,
    make_sampler_options,
    make_sampler_v2,
    make_simulator_options,
)

from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime.options.utils import Unset


class TestRuntimeSimulationHelpers(QiskitMachineLearningTestCase):
    """Tests for deterministic Runtime test helpers."""

    def test_generic_backend_has_expected_num_qubits(self):
        """Generic backend helpers create a backend with the requested width."""
        backend = make_generic_backend(2)
        self.assertEqual(backend.num_qubits, 2)

    def test_generic_backend_is_noise_free(self):
        """Generic backend helpers disable injected noise metadata."""
        backend = make_generic_backend(2)
        noisy_backend = GenericBackendV2(num_qubits=2)
        self.assertFalse(backend._noise_info)
        self.assertTrue(noisy_backend._noise_info)

    def test_generic_backend_forwards_seed(self):
        """Layout seed is forwarded to GenericBackendV2."""
        with patch("test.utils.runtime_simulation.GenericBackendV2") as mock_backend_cls:
            make_generic_backend(3, seed=42)
            mock_backend_cls.assert_called_once_with(num_qubits=3, noise_info=False, seed=42)

    def test_generic_backend_uses_default_seed(self):
        """Default layout seed matches DEFAULT_RUNTIME_SEED."""
        with patch("test.utils.runtime_simulation.GenericBackendV2") as mock_backend_cls:
            make_generic_backend(2)
            mock_backend_cls.assert_called_once_with(
                num_qubits=2, noise_info=False, seed=DEFAULT_RUNTIME_SEED
            )

    def test_simulator_options_use_fixed_seed(self):
        """Simulator options pin the simulator seed."""
        options = make_simulator_options()
        self.assertEqual(options.seed_simulator, DEFAULT_RUNTIME_SEED)

    def test_simulator_options_custom_seed(self):
        """Simulator options accept a custom seed."""
        options = make_simulator_options(seed=42)
        self.assertEqual(options.seed_simulator, 42)

    def test_estimator_options_are_deterministic(self):
        """Estimator options pin seeds and disable resilience mitigation."""
        options = make_estimator_options(default_shots=1024)
        self.assertEqual(options.seed_estimator, DEFAULT_RUNTIME_SEED)
        self.assertEqual(options.resilience_level, 0)
        self.assertEqual(options.default_shots, 1024)
        self.assertEqual(options.simulator.seed_simulator, DEFAULT_RUNTIME_SEED)

    def test_estimator_options_without_default_shots(self):
        """Estimator options omit default_shots when not requested."""
        options = make_estimator_options()
        self.assertIs(options.default_shots, Unset)

    def test_sampler_options_use_fixed_seed(self):
        """Sampler options pin the simulator seed."""
        options = make_sampler_options(default_shots=2048)
        self.assertEqual(options.simulator.seed_simulator, DEFAULT_RUNTIME_SEED)
        self.assertEqual(options.default_shots, 2048)

    def test_sampler_options_without_default_shots(self):
        """Sampler options omit default_shots when not requested."""
        options = make_sampler_options()
        self.assertIs(options.default_shots, Unset)

    def test_make_runtime_session(self):
        """Session helper returns a backend and session for the requested width."""
        backend, session = make_runtime_session(3, seed=DEFAULT_RUNTIME_SEED)
        self.assertEqual(backend.num_qubits, 3)
        self.assertIsNotNone(session)

    def test_make_runtime_pass_manager_is_reproducible(self):
        """Pass manager helper transpiles with a fixed transpiler seed."""
        backend = make_generic_backend(2, seed=DEFAULT_RUNTIME_SEED)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.measure_all()
        pass_manager = make_runtime_pass_manager(backend, seed=DEFAULT_RUNTIME_SEED)
        transpiled_a = pass_manager.run(circuit)
        transpiled_b = pass_manager.run(circuit)
        self.assertEqual(
            transpiled_a.layout.final_index_layout(),
            transpiled_b.layout.final_index_layout(),
        )

    def test_make_estimator_v2(self):
        """Estimator helper wires deterministic options and optional default shots."""
        backend, estimator = make_estimator_v2(2, seed=DEFAULT_RUNTIME_SEED, default_shots=512)
        self.assertEqual(backend.num_qubits, 2)
        self.assertEqual(estimator.options.seed_estimator, DEFAULT_RUNTIME_SEED)
        self.assertEqual(estimator.options.resilience_level, 0)
        self.assertEqual(estimator.options.default_shots, 512)
        self.assertEqual(
            estimator.options.simulator.seed_simulator,
            DEFAULT_RUNTIME_SEED,
        )

    def test_make_sampler_v2(self):
        """Sampler helper wires deterministic options and optional default shots."""
        backend, session, sampler = make_sampler_v2(
            4, seed=DEFAULT_RUNTIME_SEED, default_shots=256
        )
        self.assertEqual(backend.num_qubits, 4)
        self.assertIsNotNone(session)
        self.assertEqual(sampler.options.simulator.seed_simulator, DEFAULT_RUNTIME_SEED)
        self.assertEqual(sampler.options.default_shots, 256)
