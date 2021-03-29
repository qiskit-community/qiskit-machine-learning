# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Code inside the test is the machine learning sample from the readme.
If this test fails and code changes are needed here to resolve
the issue then ensure changes are made to readme too.
"""

import unittest
from test import QiskitMachineLearningTestCase

import numpy as np


class TestReadmeSample(QiskitMachineLearningTestCase):
    """Test sample code from readme"""

    def test_readme_sample(self):
        """ readme sample test """
        # pylint: disable=import-outside-toplevel,redefined-builtin

        def print(*args):
            """ overloads print to log values """
            if args:
                self.log.debug(args[0], *args[1:])

        # --- Exact copy of sample code ----------------------------------------

        from qiskit import BasicAer
        from qiskit.utils import QuantumInstance, algorithm_globals
        from qiskit.algorithms.optimizers import COBYLA
        from qiskit.circuit.library import TwoLocal
        from qiskit_machine_learning.algorithms import VQC
        from qiskit_machine_learning.datasets import wine
        from qiskit_machine_learning.circuit.library import RawFeatureVector

        seed = 1376
        algorithm_globals.random_seed = seed

        # Use Wine data set for training and test data
        feature_dim = 4  # dimension of each data point
        # sample_train, training_input, test_input, class_labels
        training_size = 12
        test_size = 4
        _, training_input, test_input, _ = wine(training_size=training_size,
                                                test_size=test_size,
                                                n=feature_dim)

        # prepare features and labels
        # TODO: make it better or move to wine()
        train_features = np.concatenate(list(training_input.values()))
        # one hot
        train_labels = np.concatenate([np.array([[1, 0, 0]] * training_size),
                                       np.array([[0, 1, 0]] * training_size),
                                       np.array([[0, 0, 1]] * training_size)])
        test_features = np.concatenate(list(test_input.values()))
        # one hot
        test_labels = np.concatenate([np.array([[1, 0, 0]] * test_size),
                                      np.array([[0, 1, 0]] * test_size),
                                      np.array([[0, 0, 1]] * test_size)])

        feature_map = RawFeatureVector(feature_dimension=feature_dim)
        vqc = VQC(feature_map=feature_map,
                  var_form=TwoLocal(feature_map.num_qubits, ['ry', 'rz'], 'cz', reps=3),
                  optimizer=COBYLA(maxiter=100),
                  quantum_instance=QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                                   shots=1024,
                                                   seed_simulator=seed,
                                                   seed_transpiler=seed)
                  )
        vqc.fit(train_features, train_labels)
        score = vqc.score(test_features, test_labels)
        print('Testing accuracy: {:0.2f}'.format(score))

        # ----------------------------------------------------------------------

        # self.assertGreater(result['testing_accuracy'], 0.8)


if __name__ == '__main__':
    unittest.main()
