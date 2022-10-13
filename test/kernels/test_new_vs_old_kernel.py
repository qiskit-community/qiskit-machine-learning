# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Test primitive-based quantum kernel versus the original implementation."""

from __future__ import annotations

import itertools

from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import ddt, idata, unpack
from qiskit import BasicAer
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
from qiskit.utils import algorithm_globals, QuantumInstance

from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.kernels import FidelityQuantumKernel


@ddt
class TestNewVsOldQuantumKernel(QiskitMachineLearningTestCase):
    """
    Test new quantum kernel versus the old one. The old one is evaluated on a statevector
    simulator. To be removed when old quantum kernel is removed.
    """

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 10598

        self.statevector_simulator = QuantumInstance(BasicAer.get_backend("statevector_simulator"))
        self.properties = dict(
            z1=ZFeatureMap(1),
            z2=ZFeatureMap(2),
            zz2=ZZFeatureMap(2),
            z4=ZFeatureMap(4),
            zz4=ZZFeatureMap(4),
        )

    @idata(
        itertools.product(
            ["z1", "z2", "zz2", "z4", "zz4"],
            [True, False],
            ["none", "off_diagonal", "all"],
        )
    )
    @unpack
    def test_new_vs_old(self, feature_map_name, enforce_psd, duplicates):
        """Test new versus old."""
        feature_map = self.properties[feature_map_name]
        features = algorithm_globals.random.random((10, feature_map.num_qubits))
        # add some duplicates
        features = np.concatenate((features, features[0, :].reshape(1, -1)))

        new_qk = FidelityQuantumKernel(
            feature_map=feature_map,
            enforce_psd=enforce_psd,
            evaluate_duplicates=duplicates,
        )
        old_qk = QuantumKernel(
            feature_map,
            enforce_psd=enforce_psd,
            quantum_instance=self.statevector_simulator,
            evaluate_duplicates=duplicates,
        )

        new_matrix = new_qk.evaluate(features)
        old_matrix = old_qk.evaluate(features)

        np.testing.assert_almost_equal(new_matrix, old_matrix)

        # test asymmetric case
        unseen_features = algorithm_globals.random.random((5, feature_map.num_qubits))
        # add some duplicates from the seen features
        unseen_features = np.concatenate((unseen_features, features[0, :].reshape(1, -1)))

        new_matrix = new_qk.evaluate(features, unseen_features)
        old_matrix = old_qk.evaluate(features, unseen_features)

        np.testing.assert_almost_equal(new_matrix, old_matrix)
