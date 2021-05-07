# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
gaussian dataset
"""

import numpy as np
from qiskit.utils import algorithm_globals
from qiskit.exceptions import MissingOptionalLibraryError
from .dataset_helper import features_and_labels_transform


def gaussian(training_size, test_size, n, plot_data=False, one_hot=True):
    """returns gaussian dataset"""
    sigma = 1
    if n == 2:
        class_labels = [r"A", r"B"]
        label_train = np.zeros(2 * (training_size + test_size))
        sample_train = []
        sample_a = [[0 for x in range(n)] for y in range(training_size + test_size)]
        sample_b = [[0 for x in range(n)] for y in range(training_size + test_size)]
        randomized_vector1 = algorithm_globals.random.integers(2, size=n)
        randomized_vector2 = (randomized_vector1 + 1) % 2
        for t_r in range(training_size + test_size):
            for feat in range(n):
                if randomized_vector1[feat] == 0:
                    sample_a[t_r][feat] = algorithm_globals.random.normal(-1 / 2, sigma, None)
                elif randomized_vector1[feat] == 1:
                    sample_a[t_r][feat] = algorithm_globals.random.normal(1 / 2, sigma, None)

                if randomized_vector2[feat] == 0:
                    sample_b[t_r][feat] = algorithm_globals.random.normal(-1 / 2, sigma, None)
                elif randomized_vector2[feat] == 1:
                    sample_b[t_r][feat] = algorithm_globals.random.normal(1 / 2, sigma, None)

        sample_train = [sample_a, sample_b]
        for lindex in range(training_size + test_size):
            label_train[lindex] = 0
        for lindex in range(training_size + test_size):
            label_train[training_size + test_size + lindex] = 1
        label_train = label_train.astype(int)
        sample_train = np.reshape(sample_train, (2 * (training_size + test_size), n))
        training_input = {
            key: (sample_train[label_train == k, :])[:training_size]
            for k, key in enumerate(class_labels)
        }
        test_input = {
            key: (sample_train[label_train == k, :])[training_size : (training_size + test_size)]
            for k, key in enumerate(class_labels)
        }

        training_feature_array, training_label_array = features_and_labels_transform(
            training_input, class_labels, one_hot
        )
        test_feature_array, test_label_array = features_and_labels_transform(
            test_input, class_labels, one_hot
        )

        if plot_data:
            try:
                import matplotlib.pyplot as plt
            except ImportError as ex:
                raise MissingOptionalLibraryError(
                    libname="Matplotlib",
                    name="gaussian",
                    pip_install="pip install matplotlib",
                ) from ex

            for k in range(0, 2):
                plt.scatter(
                    sample_train[label_train == k, 0][:training_size],
                    sample_train[label_train == k, 1][:training_size],
                )

            plt.title("Gaussians")
            plt.show()

        return (
            training_feature_array,
            training_label_array,
            test_feature_array,
            test_label_array,
        )
    elif n == 3:
        class_labels = [r"A", r"B", r"C"]
        label_train = np.zeros(3 * (training_size + test_size))
        sample_train = []
        sample_a = [[0 for x in range(n)] for y in range(training_size + test_size)]
        sample_b = [[0 for x in range(n)] for y in range(training_size + test_size)]
        sample_c = [[0 for x in range(n)] for y in range(training_size + test_size)]
        randomized_vector1 = algorithm_globals.random.integers(3, size=n)
        randomized_vector2 = (randomized_vector1 + 1) % 3
        randomized_vector3 = (randomized_vector2 + 1) % 3
        for t_r in range(training_size + test_size):
            for feat in range(n):
                if randomized_vector1[feat] == 0:
                    sample_a[t_r][feat] = algorithm_globals.random.normal(
                        2 * 1 * np.pi / 6, sigma, None
                    )
                elif randomized_vector1[feat] == 1:
                    sample_a[t_r][feat] = algorithm_globals.random.normal(
                        2 * 3 * np.pi / 6, sigma, None
                    )
                elif randomized_vector1[feat] == 2:
                    sample_a[t_r][feat] = algorithm_globals.random.normal(
                        2 * 5 * np.pi / 6, sigma, None
                    )

                if randomized_vector2[feat] == 0:
                    sample_b[t_r][feat] = algorithm_globals.random.normal(
                        2 * 1 * np.pi / 6, sigma, None
                    )
                elif randomized_vector2[feat] == 1:
                    sample_b[t_r][feat] = algorithm_globals.random.normal(
                        2 * 3 * np.pi / 6, sigma, None
                    )
                elif randomized_vector2[feat] == 2:
                    sample_b[t_r][feat] = algorithm_globals.random.normal(
                        2 * 5 * np.pi / 6, sigma, None
                    )

                if randomized_vector3[feat] == 0:
                    sample_c[t_r][feat] = algorithm_globals.random.normal(
                        2 * 1 * np.pi / 6, sigma, None
                    )
                elif randomized_vector3[feat] == 1:
                    sample_c[t_r][feat] = algorithm_globals.random.normal(
                        2 * 3 * np.pi / 6, sigma, None
                    )
                elif randomized_vector3[feat] == 2:
                    sample_c[t_r][feat] = algorithm_globals.random.normal(
                        2 * 5 * np.pi / 6, sigma, None
                    )

        sample_train = [sample_a, sample_b, sample_c]
        for lindex in range(training_size + test_size):
            label_train[lindex] = 0
        for lindex in range(training_size + test_size):
            label_train[training_size + test_size + lindex] = 1
        for lindex in range(training_size + test_size):
            label_train[training_size + test_size + training_size + test_size + lindex] = 2
        label_train = label_train.astype(int)
        sample_train = np.reshape(sample_train, (3 * (training_size + test_size), n))
        training_input = {
            key: (sample_train[label_train == k, :])[:training_size]
            for k, key in enumerate(class_labels)
        }
        test_input = {
            key: (sample_train[label_train == k, :])[training_size : (training_size + test_size)]
            for k, key in enumerate(class_labels)
        }

        training_feature_array, training_label_array = features_and_labels_transform(
            training_input, class_labels, one_hot
        )
        test_feature_array, test_label_array = features_and_labels_transform(
            test_input, class_labels, one_hot
        )

        if plot_data:
            try:
                import matplotlib.pyplot as plt
            except ImportError as ex:
                raise MissingOptionalLibraryError(
                    libname="Matplotlib",
                    name="gaussian",
                    pip_install="pip install matplotlib",
                ) from ex

            for k in range(0, 3):
                plt.scatter(
                    sample_train[label_train == k, 0][:training_size],
                    sample_train[label_train == k, 1][:training_size],
                )

            plt.title("Gaussians")
            plt.show()

        return (
            training_feature_array,
            training_label_array,
            test_feature_array,
            test_label_array,
        )
    else:
        raise ValueError("Gaussian presently only supports 2 or 3 qubits")
