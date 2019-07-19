# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import operator

import numpy as np
from sklearn.decomposition import PCA


def get_num_classes(dataset):
    """Check number of classes in a given dataset

    Args:
        dataset(dict): key is the class name and value is the data.

    Returns:
        int: number of classes
    """
    return len(list(dataset.keys()))


def get_feature_dimension(dataset):
    """Check feature dimension of a given dataset

    Args:
        dataset(dict): key is the class name and value is the data.

    Returns:
        int: feature dimension, -1 denotes no data in the dataset.
    """
    if not isinstance(dataset, dict):
        raise TypeError("Dataset is not formatted as a dict. Please check it.")

    feature_dim = -1
    for v in dataset.values():
        if not isinstance(v, np.ndarray):
            v = np.asarray(v)
        return v.shape[1]

    return feature_dim


def split_dataset_to_data_and_labels(dataset, class_names=None):
    """Split dataset to data and labels numpy array

        If `class_names` is given, use the desired label to class name mapping,
        or create the mapping based on the keys in the dataset.

    Args:
        dataset (dict): {'A': numpy.ndarray, 'B': numpy.ndarray, ...}
        class_names (dict): class name of dataset, {class_name: label}

    Returns:
        [numpy.ndarray, numpy.ndarray]: idx 0 is data, NxD array,
                    idx 1 is labels, Nx1 array, value is ranged
                    from 0 to K-1, K is the number of classes
        dict: {str: int}, map from class name to label
    """
    data = []
    labels = []
    if class_names is None:
        sorted_classes_name = sorted(list(dataset.keys()))
        class_to_label = {k: idx for idx, k in enumerate(sorted_classes_name)}
    else:
        class_to_label = class_names
    sorted_label = sorted(class_to_label.items(), key=operator.itemgetter(1))
    for class_name, label in sorted_label:
        values = dataset[class_name]
        for value in values:
            data.append(value)
            try:
                labels.append(class_to_label[class_name])
            except Exception as e:
                raise KeyError('The dataset has different class names to '
                               'the training data. error message: {}'.format(e))
    data = np.asarray(data)
    labels = np.asarray(labels)
    if class_names is None:
        return [data, labels], class_to_label
    else:
        return [data, labels]


def map_label_to_class_name(predicted_labels, label_to_class):
    """Helper converts labels (numeric) to class name (string)

    Args:
        predicted_labels (numpy.ndarray): Nx1 array
        label_to_class (dict or list): a mapping form label (numeric) to class name (str)
    Returns:
        [str]: predicted class names of each datum
    """

    if not isinstance(predicted_labels, np.ndarray):
        predicted_labels = np.asarray([predicted_labels])

    predicted_class_names = []

    for predicted_label in predicted_labels:
        predicted_class_names.append(label_to_class[predicted_label])
    return predicted_class_names


def reduce_dim_to_via_pca(x, dim):
    """
    Reduce the data dimension via pca

    Args:
        x (numpy.ndarray): NxD array
        dim (int): the targeted dimension D'

    Returns:
        numpy.ndarray: NxD' array

    """
    x_reduced = PCA(n_components=dim).fit_transform(x)
    return x_reduced
