# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
ad hoc dataset
"""

import itertools as it
from functools import reduce
import numpy as np
import scipy
from qiskit.utils import algorithm_globals
from qiskit.utils.optionals import HAS_MATPLOTLIB

from qiskit_machine_learning.datasets.dataset_helper import (
    features_and_labels_transform,
)


def ad_hoc_data(
    training_size,
    test_size,
    n,
    gap,
    plot_data=False,
    one_hot=True,
    include_sample_total=False,
):
    """returns ad hoc dataset"""
    class_labels = [r"A", r"B"]
    count = 0
    if n == 2:
        count = 100
    elif n == 3:
        count = 20  # coarseness of data separation
    else:
        raise ValueError(f"Supported values of 'n' are 2 and 3 only, but {n} is provided.")

    # Define auxiliary matrices and initial state
    z = np.diag([1, -1])
    i_2 = np.eye(2)
    h_2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    h_n = reduce(np.kron, [h_2] * n)
    psi_0 = np.ones(2 ** n) / np.sqrt(2 ** n)

    # Generate Z matrices acting on each qubits
    z_i = np.array([reduce(np.kron, [i_2] * i + [z] + [i_2] * (n - i - 1)) for i in range(n)])

    # Construct the parity operator
    bitstrings = ["".join(bstring) for bstring in it.product(*[["0", "1"]] * n)]
    if n == 2:
        bitstring_parity = [bstr.count("1") % 2 for bstr in bitstrings]
        d_m = np.diag((-1) ** np.array(bitstring_parity))
    elif n == 3:
        bitstring_majority = [0 if bstr.count("0") > 1 else 1 for bstr in bitstrings]
        d_m = np.diag((-1) ** np.array(bitstring_majority))

    # Generate a random unitary operator
    basis = algorithm_globals.random.random(
        (2 ** n, 2 ** n)
    ) + 1j * algorithm_globals.random.random((2 ** n, 2 ** n))
    basis = np.array(basis).conj().T @ np.array(basis)
    eigvals, eigvecs = np.linalg.eig(basis)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    m_m = eigvecs.conj().T @ d_m @ eigvecs

    # Compute expectation value of parity in grid
    xvals = np.linspace(0, 2 * np.pi, count, endpoint=False)
    ind_pairs = list(it.combinations(range(n), 2))
    sample_total = []
    for x in it.product(*[xvals] * n):
        x = np.array(x)
        phi = np.sum(x[:, None, None] * z_i, axis=0)
        phi += sum([(np.pi - x[i1]) * (np.pi - x[i2]) * z_i[i1] @ z_i[i2] for i1, i2 in ind_pairs])
        u_u = scipy.linalg.expm(1j * phi)  # pylint: disable=no-member
        psi = u_u @ h_n @ u_u @ psi_0
        exp_val = np.real(psi.conj().T @ m_m @ psi)
        if np.abs(exp_val) > gap:
            sample_total.append(np.sign(exp_val))
        else:
            sample_total.append(0)
    sample_total = np.array(sample_total).reshape(*[count] * n)

    # Extract samples from grid
    x_sample, y_sample = _sample_ad_hoc_data(sample_total, xvals, training_size + test_size, n)

    if plot_data:
        _plot_ad_hoc_data(x_sample, y_sample, training_size)

    training_input = {
        key: (x_sample[y_sample == k, :])[:training_size] for k, key in enumerate(class_labels)
    }
    test_input = {
        key: (x_sample[y_sample == k, :])[training_size : (training_size + test_size)]
        for k, key in enumerate(class_labels)
    }

    training_feature_array, training_label_array = features_and_labels_transform(
        training_input, class_labels, one_hot
    )
    test_feature_array, test_label_array = features_and_labels_transform(
        test_input, class_labels, one_hot
    )

    if include_sample_total:
        return (
            training_feature_array,
            training_label_array,
            test_feature_array,
            test_label_array,
            sample_total,
        )
    else:
        return (
            training_feature_array,
            training_label_array,
            test_feature_array,
            test_label_array,
        )


def _sample_ad_hoc_data(sample_total, xvals, num_samples, n):
    count = sample_total.shape[0]
    sample_a, sample_b = [], []
    for i, sample_list in enumerate([sample_a, sample_b]):
        label = 1 if i == 0 else -1
        while len(sample_list) < num_samples:
            draws = tuple(algorithm_globals.random.choice(count) for i in range(n))
            if sample_total[draws] == label:
                sample_list.append([xvals[d] for d in draws])

    labels = np.array([0] * num_samples + [1] * num_samples)
    samples = [sample_a, sample_b]
    samples = np.reshape(samples, (2 * num_samples, n))
    return samples, labels


@HAS_MATPLOTLIB.require_in_call
def _plot_ad_hoc_data(x_total, y_total, training_size):
    import matplotlib.pyplot as plt
    n = x_total.shape[1]
    fig = plt.figure()
    projection = "3d" if n == 3 else None
    ax1 = fig.add_subplot(1, 1, 1, projection=projection)
    for k in range(0, 2):
        ax1.scatter(*x_total[y_total == k][:training_size].T)
    ax1.set_title("Ad-hoc Data")
    plt.show()
