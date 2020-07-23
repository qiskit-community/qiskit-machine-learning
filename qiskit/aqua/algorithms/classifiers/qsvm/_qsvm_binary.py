# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The binary classifier."""

import logging
import warnings

import numpy as np

from qiskit.aqua.utils import map_label_to_class_name, optimize_svm
from ._qsvm_abc import _QSVM_ABC

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class _QSVM_Binary(_QSVM_ABC):
    """The binary classifier."""

    def construct_circuit(self, x1, x2, measurement=False):
        """ construct circuit """
        warnings.warn("Please use the 'construct_circuit' in the qsvm class directly.",
                      DeprecationWarning)
        return self._qalgo.construct_circuit(x1, x2, measurement)

    def construct_kernel_matrix(self, x1_vec, x2_vec=None):
        """ construct kernel matrix """
        warnings.warn("Please use the 'construct_kernel_matrix' in the qsvm "
                      "class directly.", DeprecationWarning)
        return self._qalgo.construct_kernel_matrix(x1_vec, x2_vec, self._qalgo.quantum_instance)

    def get_predicted_confidence(self, data, return_kernel_matrix=False):
        """Get predicted confidence.

        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
            return_kernel_matrix (bool): return kernel matrix
        Returns:
            numpy.ndarray: Nx1 array, predicted confidence
            numpy.ndarray (optional): the kernel matrix, NxN1, where N1 is
                                      the number of support vectors.
        """
        alphas = self._ret['svm']['alphas']
        bias = self._ret['svm']['bias']
        svms = self._ret['svm']['support_vectors']
        yin = self._ret['svm']['yin']
        kernel_matrix = self._qalgo.construct_kernel_matrix(data, svms)

        confidence = np.sum(yin * alphas * kernel_matrix, axis=1) + bias

        if return_kernel_matrix:
            return confidence, kernel_matrix
        else:
            return confidence

    def train(self, data, labels):
        """
        Train the svm.

        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
            labels (numpy.ndarray): Nx1 array, where N is the number of data
        """
        scaling = 1.0 if self._qalgo.quantum_instance.is_statevector else None
        kernel_matrix = self._qalgo.construct_kernel_matrix(data)
        labels = labels * 2 - 1  # map label from 0 --> -1 and 1 --> 1
        labels = labels.astype(np.float)
        [alpha, b, support] = optimize_svm(kernel_matrix, labels, scaling=scaling)
        support_index = np.where(support)
        alphas = alpha[support_index]
        svms = data[support_index]
        yin = labels[support_index]

        self._ret['kernel_matrix_training'] = kernel_matrix
        self._ret['svm'] = {}
        self._ret['svm']['alphas'] = alphas
        self._ret['svm']['bias'] = b
        self._ret['svm']['support_vectors'] = svms
        self._ret['svm']['yin'] = yin

    def test(self, data, labels):
        """
        Test the svm.

        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
            labels (numpy.ndarray): Nx1 array, where N is the number of data

        Returns:
            float: accuracy
        """
        predicted_confidence, kernel_matrix = self.get_predicted_confidence(data, True)
        binarized_predictions = (np.sign(predicted_confidence) + 1) / 2  # remap -1 --> 0, 1 --> 1
        predicted_labels = binarized_predictions.astype(int)
        accuracy = np.sum(predicted_labels == labels.astype(int)) / labels.shape[0]
        logger.debug("Classification success for this set is {:.2f}% \n".format(accuracy * 100.0))
        self._ret['kernel_matrix_testing'] = kernel_matrix
        self._ret['testing_accuracy'] = accuracy
        # test_success_ratio is deprecated
        self._ret['test_success_ratio'] = accuracy
        return accuracy

    def predict(self, data):
        """
        Predict using the svm.

        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
        Returns:
            numpy.ndarray: predicted labels, Nx1 array
        """
        predicted_confidence = self.get_predicted_confidence(data)
        binarized_predictions = (np.sign(predicted_confidence) + 1) / 2  # remap -1 --> 0, 1 --> 1
        predicted_labels = binarized_predictions.astype(int)
        return predicted_labels

    def run(self):
        """Put the train, test, predict together."""
        self.train(self._qalgo.training_dataset[0], self._qalgo.training_dataset[1])
        if self._qalgo.test_dataset is not None:
            self.test(self._qalgo.test_dataset[0], self._qalgo.test_dataset[1])
        if self._qalgo.datapoints is not None:
            predicted_labels = self.predict(self._qalgo.datapoints)
            predicted_classes = map_label_to_class_name(predicted_labels,
                                                        self._qalgo.label_to_class)
            self._ret['predicted_labels'] = predicted_labels
            self._ret['predicted_classes'] = predicted_classes

        return self._ret

    def load_model(self, file_path):
        """ load model """
        model_npz = np.load(file_path, allow_pickle=True)  # pylint: disable=unexpected-keyword-arg
        model = {'alphas': model_npz['alphas'],
                 'bias': model_npz['bias'],
                 'support_vectors': model_npz['support_vectors'],
                 'yin': model_npz['yin']}
        self._ret['svm'] = model
        try:
            self._qalgo.class_to_label = model_npz['class_to_label']
            self._qalgo.label_to_class = model_npz['label_to_class']
        except KeyError as ex:
            logger.warning("The model saved in Aqua 0.5 does not contain the mapping "
                           "between class names and labels. "
                           "Please setup them and save the model again "
                           "for further use. Error: %s", str(ex))

    def save_model(self, file_path):
        """ save model """
        model = {'alphas': self._ret['svm']['alphas'],
                 'bias': self._ret['svm']['bias'],
                 'support_vectors': self._ret['svm']['support_vectors'],
                 'yin': self._ret['svm']['yin'],
                 'class_to_label': self._qalgo.class_to_label,
                 'label_to_class': self._qalgo.label_to_class}
        np.savez(file_path, **model)
