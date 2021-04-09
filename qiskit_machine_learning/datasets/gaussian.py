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
    """ returns gaussian dataset """
    if n not in (2, 3):
        raise ValueError("Gaussian presently only supports 2 or 3 qubits")

    class_labels = _generate_class_labels(n)
    label_train, randomized_vectors, samples_train = _init_data_structures(n, test_size,
                                                                           training_size)
    sigma = 1

    for t_r in range(training_size + test_size):
        for feat in range(n):
            for sample, randomized_vector in zip(samples_train, randomized_vectors):
                sample[t_r][feat] = _calc_random_normal(randomized_vector, feat,
                                                        sigma) if n == 2 \
                    else _calc_random_normal_2(randomized_vector, feat, sigma)
                # TODO unify both functions in the future when other n's supported

    _update_label_train(n, label_train, test_size, training_size)
    label_train = label_train.astype(int)

    samples_train = np.reshape(samples_train, (n * (training_size + test_size), n))
    training_input = {key: (samples_train[label_train == k, :])[:training_size]
                      for k, key in enumerate(class_labels)}
    test_input = {
        key: (samples_train[label_train == k, :])[training_size:(training_size + test_size)] for
        k, key in enumerate(class_labels)}

    training_feature_array, training_label_array = features_and_labels_transform(
        training_input, class_labels, one_hot)
    test_feature_array, test_label_array = features_and_labels_transform(
        test_input, class_labels, one_hot)

    if plot_data:
        _plot(n, label_train, samples_train, training_size)

    return training_feature_array, training_label_array, test_feature_array, test_label_array


def _generate_class_labels(n):
    if n > 25:
        # TODO change to it a more complex labeling system e.g. AA, AB, AAA etc. or just to C1,
        #  C2, C3 etc.
        raise Exception("To many classes requested. Maximum is 26")
    capital_a_ascii = 65
    return [r'%c' % x for x in range(capital_a_ascii, capital_a_ascii + n)]


def _update_label_train(n, label_train, test_size, training_size):
    for lindex in range(training_size + test_size):
        for ind in range(n):
            label_train[ind * (training_size + test_size) + lindex] = ind


def _plot(n, label_train, sample_train, training_size):
    try:
        import matplotlib.pyplot as plt
    except ImportError as ex:
        raise MissingOptionalLibraryError(
            libname='Matplotlib',
            name='gaussian',
            pip_install='pip install matplotlib') from ex
    for k in range(n):
        plt.scatter(sample_train[label_train == k, 0][:training_size],
                    sample_train[label_train == k, 1][:training_size])
    plt.title("Gaussians")
    plt.show()


def _init_data_structures(n, test_size, training_size):
    label_train = np.zeros(n * (training_size + test_size))
    samples = _init_samples(n, test_size, training_size)
    randomized_vectors = _init_randomized_vectors(n)
    return label_train, randomized_vectors, samples


def _init_randomized_vectors(n):
    randomized_vector = algorithm_globals.random.integers(n, size=n)
    randomized_vectors = [randomized_vector]
    for _ in range(n - 1):
        randomized_vector = (randomized_vector + 1) % n
        randomized_vectors.append(randomized_vector)
    return randomized_vectors


def _init_samples(n, test_size, training_size):
    samples = []
    for _ in range(n):
        sample = [[0 for _ in range(n)] for _ in range(training_size + test_size)]
        samples.append(sample)
    return samples


def _calc_random_normal(randomized_vector, feat, sigma):
    center = 1 / 2
    return algorithm_globals.random.normal(pow(-1, randomized_vector[feat]) * center, sigma, None)


def _calc_random_normal_2(randomized_vector, feat, sigma):
    coefficient = 2 * (1 + 2 * randomized_vector[feat])
    return algorithm_globals.random.normal(coefficient * np.pi / 6, sigma, None)
