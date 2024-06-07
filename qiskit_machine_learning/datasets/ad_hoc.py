# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2024.
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
from __future__ import annotations

import itertools as it
from functools import reduce
from typing import Tuple, Dict, List

import numpy as np
from qiskit.utils import optionals
from qiskit_algorithms.utils import algorithm_globals
from sklearn import preprocessing


def ad_hoc_data(
    training_size: int,
    test_size: int,
    n: int,
    gap: int,
    plot_data: bool = False,
    one_hot: bool = True,
    include_sample_total: bool = False,
) -> (
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    r"""Generates a toy dataset that can be fully separated with
    :class:`~qiskit.circuit.library.ZZFeatureMap` according to the procedure
    outlined in [1]. To construct the dataset, we first sample uniformly
    distributed vectors :math:`\vec{x} \in (0, 2\pi]^{n}` and apply the
    feature map

    .. math::
        |\Phi(\vec{x})\rangle = U_{{\Phi} (\vec{x})} H^{\otimes n} U_{{\Phi} (\vec{x})}
        H^{\otimes n} |0^{\otimes n} \rangle

    where

    .. math::
        U_{{\Phi} (\vec{x})} = \exp \left( i \sum_{S \subseteq [n] } \phi_S(\vec{x})
        \prod_{i \in S} Z_i \right)

    and

    .. math::
        \begin{cases}
        \phi_{\{i, j\}} = (\pi - x_i)(\pi - x_j) \\
        \phi_{\{i\}} = x_i
        \end{cases}

    We then attribute labels to the vectors according to the rule

    .. math::
        m(\vec{x}) = \begin{cases}
        1 & \langle \Phi(\vec{x}) | V^\dagger \prod_i Z_i V | \Phi(\vec{x}) \rangle > \Delta \\
        -1 & \langle \Phi(\vec{x}) | V^\dagger \prod_i Z_i V | \Phi(\vec{x}) \rangle < -\Delta
        \end{cases}

    where :math:`\Delta` is the separation gap, and
    :math:`V\in \mathrm{SU}(4)` is a random unitary.

    The current implementation only works with n = 2 or 3.

    **References:**

    [1] Havlíček V, Córcoles AD, Temme K, Harrow AW, Kandala A, Chow JM,
    Gambetta JM. Supervised learning with quantum-enhanced feature
    spaces. Nature. 2019 Mar;567(7747):209-12.
    `arXiv:1804.11326 <https://arxiv.org/abs/1804.11326>`_

    Args:
        training_size: the number of training samples.
        test_size: the number of testing samples.
        n: number of qubits (dimension of the feature space). Must be 2 or 3.
        gap: separation gap (:math:`\Delta`).
        plot_data: whether to plot the data. Requires matplotlib.
        one_hot: if True, return the data in one-hot format.
        include_sample_total: if True, return all points in the uniform
            grid in addition to training and testing samples.

    Returns:
        Training and testing samples.

    Raises:
        ValueError: if n is not 2 or 3.
    """
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
    psi_0 = np.ones(2**n) / np.sqrt(2**n)

    # Generate Z matrices acting on each qubits
    z_i = np.array([reduce(np.kron, [i_2] * i + [z] + [i_2] * (n - i - 1)) for i in range(n)])

    # Construct the parity operator
    bitstrings = ["".join(bstring) for bstring in it.product(*[["0", "1"]] * n)]
    if n == 2:
        bitstring_parity = [bstr.count("1") % 2 for bstr in bitstrings]
        d_m = np.diag((-1) ** np.array(bitstring_parity))
    else:  # n must be 3 here, as n checked above which allows only 2 and 3
        bitstring_majority = [0 if bstr.count("0") > 1 else 1 for bstr in bitstrings]
        d_m = np.diag((-1) ** np.array(bitstring_majority))

    # Generate a random unitary operator by collecting eigenvectors of a
    # random hermitian operator
    basis = algorithm_globals.random.random(
        (2**n, 2**n)
    ) + 1j * algorithm_globals.random.random((2**n, 2**n))
    basis = np.array(basis).conj().T @ np.array(basis)
    eigvals, eigvecs = np.linalg.eig(basis)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    m_m = eigvecs.conj().T @ d_m @ eigvecs

    # Generate a grid of points in the feature space and compute the
    # expectation value of the parity
    xvals = np.linspace(0, 2 * np.pi, count, endpoint=False)
    ind_pairs = list(it.combinations(range(n), 2))
    _sample_total = []
    for x in it.product(*[xvals] * n):
        x_arr = np.array(x)
        phi = np.sum(x_arr[:, None, None] * z_i, axis=0)
        phi += sum(
            ((np.pi - x_arr[i1]) * (np.pi - x_arr[i2]) * z_i[i1] @ z_i[i2] for i1, i2 in ind_pairs)
        )
        # u_u was actually scipy.linalg.expm(1j * phi), but this method is
        # faster because phi is always a diagonal matrix.
        # We first extract the diagonal elements, then do exponentiation, then
        # construct a diagonal matrix from them.
        u_u = np.diag(np.exp(1j * np.diag(phi)))
        psi = u_u @ h_n @ u_u @ psi_0
        exp_val = np.real(psi.conj().T @ m_m @ psi)
        if np.abs(exp_val) > gap:
            _sample_total.append(np.sign(exp_val))
        else:
            _sample_total.append(0)
    sample_total = np.array(_sample_total).reshape(*[count] * n)

    # Extract training and testing samples from grid
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

    training_feature_array, training_label_array = _features_and_labels_transform(
        training_input, class_labels, one_hot
    )
    test_feature_array, test_label_array = _features_and_labels_transform(
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


@optionals.HAS_MATPLOTLIB.require_in_call
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


def _features_and_labels_transform(
    dataset: Dict[str, np.ndarray], class_labels: List[str], one_hot: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a dataset into arrays of features and labels.

    Args:
        dataset: A dictionary in the format of {'A': numpy.ndarray, 'B': numpy.ndarray, ...}
        class_labels: A list of classes in the dataset
        one_hot (bool): if True - return one-hot encoded label

    Returns:
        A tuple of features as np.ndarray, label as np.ndarray
    """
    features = np.concatenate(list(dataset.values()))

    raw_labels = []
    for category in dataset.keys():
        num_samples = dataset[category].shape[0]
        raw_labels += [category] * num_samples

    if not raw_labels:
        # no labels, empty dataset
        labels = np.zeros((0, len(class_labels)))
        return features, labels

    if one_hot:
        encoder = preprocessing.OneHotEncoder()
        encoder.fit(np.array(class_labels).reshape(-1, 1))
        labels = encoder.transform(np.array(raw_labels).reshape(-1, 1))
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels.todense())
    else:
        encoder = preprocessing.LabelEncoder()
        encoder.fit(np.array(class_labels))
        labels = encoder.transform(np.array(raw_labels))

    return features, labels
