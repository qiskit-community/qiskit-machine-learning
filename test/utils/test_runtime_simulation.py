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

from test import QiskitMachineLearningTestCase
from test.utils.runtime_simulation import (
    DEFAULT_RUNTIME_SEED,
    make_estimator_options,
    make_generic_backend,
    make_sampler_options,
    make_simulator_options,
)


class TestRuntimeSimulationHelpers(QiskitMachineLearningTestCase):
    """Tests for deterministic Runtime test helpers."""

    def test_backend_uses_fixed_seed(self):
        """Generic backend helpers create a noise-free fake backend."""
        backend = make_generic_backend(2)
        self.assertEqual(backend.num_qubits, 2)

    def test_simulator_options_use_fixed_seed(self):
        """Simulator options pin the simulator seed."""
        options = make_simulator_options()
        self.assertEqual(options.seed_simulator, DEFAULT_RUNTIME_SEED)

    def test_estimator_options_are_deterministic(self):
        """Estimator options pin seeds and disable resilience mitigation."""
        options = make_estimator_options(default_shots=1024)
        self.assertEqual(options.seed_estimator, DEFAULT_RUNTIME_SEED)
        self.assertEqual(options.resilience_level, 0)
        self.assertEqual(options.default_shots, 1024)
        self.assertEqual(options.simulator.seed_simulator, DEFAULT_RUNTIME_SEED)

    def test_sampler_options_use_fixed_seed(self):
        """Sampler options pin the simulator seed."""
        options = make_sampler_options(default_shots=2048)
        self.assertEqual(options.simulator.seed_simulator, DEFAULT_RUNTIME_SEED)
        self.assertEqual(options.default_shots, 2048)
