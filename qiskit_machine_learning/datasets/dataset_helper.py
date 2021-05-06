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

""" Data set helper """

from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
from sklearn import preprocessing


def discretize_and_truncate(
    data,
    min_max_bin_centers,
    num_qubits,
    return_data_grid_elements=False,
    return_prob=False,
    prob_non_zero=True,
):
    """
    Discretize & truncate classical data to enable digital encoding in qubit registers
    whereby the data grid is ``[[grid elements dim 0], ..., [grid elements dim k]]``.

    For each dimension ``k``, the domain is split into ``(2 ** num_qubits[k])`` bins equally spaced
    and equally sized, each centered in
    ``min_max_bin_centers[k, 0], ..., min_max_bin_centers[k, 1]``. Bins have size equal to
    ``(min_max_bin_centers[k, 1] - min_max_bin_centers[k, 0]) / (2 ** num_qubits[k] - 1)``.
    Notice that:

    * Every sample in data that falls out of the bins is discarded.
    * The leftmost bin extends both to the left and to the right around its center,

    therefore ``min_max_bin_centers[k, 0]`` is not the left bound for truncation, but only
    the center of the leftmost bin. Similar considerations hold for ``min_max_bin_centers[k, 1]``
    on the right.

    Args:
        data (list or array or np.array): training data (int or float) of dimension ``k``.
        min_max_bin_centers (list or array or np.ndarray):  ``k`` min/max data values
            ``[[min_center_0, max_center_0],...,[min_center_k-1, max_center_k-1]]``.
            If univariate data: ``[min_center_0, max_center_0]``.
        num_qubits (list or array or np.array): ``k`` numbers of qubits to determine
            representation resolution, i.e. n qubits enable the representation of 2**n
            values ``[num_qubits_0,..., num_qubits_k-1]``.
        return_data_grid_elements (Bool): if ``True`` - return an array with the data grid
            elements.
        return_prob (Bool): if ``True`` - return a normalized frequency count of the discretized and
            truncated data samples.
        prob_non_zero (Bool): if ``True`` - set 0 values in the prob_data to ``10^-1`` to avoid
            potential problems when using the probabilities in loss functions - division by 0.

    Returns:
        array: discretized and truncated data.
        array: data grid ``[[grid elements dim 0],..., [grid elements dim k]]``.
        array: grid elements, ``Product_j=0^k-1 2**num_qubits_j`` element vectors.
        array: data probability, normalized frequency count sorted from smallest to biggest element.

    """
    # Truncate the data
    if np.ndim(min_max_bin_centers) == 1:
        min_max_bin_centers = np.reshape(min_max_bin_centers, (1, len(min_max_bin_centers)))

    data = data.reshape((len(data), len(num_qubits)))
    temp = []
    for i, data_sample in enumerate(data):
        append = True
        for j, entry in enumerate(data_sample):
            if entry < min_max_bin_centers[j, 0] - 0.5 / (2 ** num_qubits[j] - 1) * (
                min_max_bin_centers[j, 1] - min_max_bin_centers[j, 0]
            ):
                append = False
            if entry > min_max_bin_centers[j, 1] + 0.5 / (2 ** num_qubits[j] - 1) * (
                min_max_bin_centers[j, 1] - min_max_bin_centers[j, 0]
            ):
                append = False
        if append:
            temp.append(list(data_sample))
    data = np.array(temp, dtype=float)

    # Fit the data to the data element grid
    for j, prec in enumerate(num_qubits):
        data_row = data[:, j]  # dim j of all data samples
        # prepare element grid for dim j
        elements_current_dim = np.linspace(
            min_max_bin_centers[j, 0], min_max_bin_centers[j, 1], (2 ** prec)
        )
        # find index for data sample in grid
        index_grid = np.searchsorted(
            elements_current_dim,
            data_row - (elements_current_dim[1] - elements_current_dim[0]) * 0.5,
        )
        for k, index in enumerate(index_grid):
            data[k, j] = elements_current_dim[index]
        if j == 0:
            if len(num_qubits) > 1:
                data_grid = [elements_current_dim]
            else:
                data_grid = elements_current_dim
            grid_elements = elements_current_dim
        elif j == 1:
            temp = []
            for grid_element in grid_elements:
                for element_current in elements_current_dim:
                    temp.append([grid_element, element_current])
            grid_elements = temp
            data_grid.append(elements_current_dim)
        else:
            temp = []
            for grid_element in grid_elements:
                for element_current in elements_current_dim:
                    temp.append(grid_element + [element_current])
            grid_elements = deepcopy(temp)
            data_grid.append(elements_current_dim)
    data_grid = np.array(data_grid, dtype=object)

    data = np.reshape(data, (len(data), len(data[0])))

    if return_prob:
        if np.ndim(data) > 1:
            prob_data = np.zeros(int(np.prod(np.power(np.ones(len(data[0])) * 2, num_qubits))))
        else:
            prob_data = np.zeros(int(np.prod(np.power(np.array([2]), num_qubits))))
        for data_element in data:
            for i, element in enumerate(grid_elements):
                if all(data_element == element):
                    prob_data[i] += 1 / len(data)
        if prob_non_zero:
            # add epsilon to avoid 0 entries which can be problematic in loss functions (division)
            prob_data = [1e-10 if x == 0 else x for x in prob_data]

        if return_data_grid_elements:
            return data, data_grid, grid_elements, prob_data
        else:
            return data, data_grid, prob_data

    else:
        if return_data_grid_elements:
            return data, data_grid, grid_elements

        else:
            return data, data_grid


def features_and_labels_transform(
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
