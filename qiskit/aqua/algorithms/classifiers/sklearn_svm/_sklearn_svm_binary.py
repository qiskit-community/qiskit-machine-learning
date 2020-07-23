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

"""
the binary classifier
"""

import logging

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from qiskit.aqua.utils import map_label_to_class_name, optimize_svm
from ._sklearn_svm_abc import _SklearnSVMABC

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class _SklearnSVMBinary(_SklearnSVMABC):
    """
    the binary classifier
    """

    def construct_kernel_matrix(self, points_array, points_array2, gamma=None):
        """ construct kernel matrix """
        return rbf_kernel(points_array, points_array2, gamma)

    def train(self, data, labels):
        """
        train the svm
        Args:
            data (dict): dictionary which maps each class to the points in the class
            labels (list): list of classes. For example: ['A', 'B']
        """
        labels = labels.astype(np.float)
        labels = labels * 2. - 1.
        kernel_matrix = self.construct_kernel_matrix(data, data, self.gamma)
        self._ret['kernel_matrix_training'] = kernel_matrix
        [alpha, b, support] = optimize_svm(kernel_matrix, labels)
        alphas = np.array([])
        svms = np.array([])
        yin = np.array([])
        for alphindex, _ in enumerate(support):
            if support[alphindex]:
                alphas = np.vstack([alphas, alpha[alphindex]]) if alphas.size else alpha[alphindex]
                svms = np.vstack([svms, data[alphindex]]) if svms.size else data[alphindex]
                yin = np.vstack([yin, labels[alphindex]]
                                ) if yin.size else labels[alphindex]

        self._ret['svm'] = {}
        self._ret['svm']['alphas'] = alphas
        self._ret['svm']['bias'] = b
        self._ret['svm']['support_vectors'] = svms
        self._ret['svm']['yin'] = yin

    def test(self, data, labels):
        """
        test the svm
        Args:
            data (dict): dictionary which maps each class to the points in the class
            labels (list): list of classes. For example: ['A', 'B']
        Returns:
            float: final success ration
        """
        alphas = self._ret['svm']['alphas']
        bias = self._ret['svm']['bias']
        svms = self._ret['svm']['support_vectors']
        yin = self._ret['svm']['yin']

        kernel_matrix = self.construct_kernel_matrix(data, svms, self.gamma)
        self._ret['kernel_matrix_testing'] = kernel_matrix

        success_ratio = 0
        _l = 0
        total_num_points = len(data)
        lsign = np.zeros(total_num_points)
        for tin in range(total_num_points):
            ltot = 0
            for sin in range(len(svms)):
                _l = yin[sin] * alphas[sin] * kernel_matrix[tin][sin]
                ltot += _l
            lsign[tin] = (np.sign(ltot + bias) + 1.) / 2.

            logger.debug("\n=============================================")
            logger.debug('classifying %s.', data[tin])
            logger.debug('Label should be %s.', self.label_to_class[np.int(labels[tin])])
            logger.debug('Predicted label is %s.', self.label_to_class[np.int(lsign[tin])])
            if np.int(labels[tin]) == np.int(lsign[tin]):
                logger.debug('CORRECT')
            else:
                logger.debug('INCORRECT')
            if lsign[tin] == labels[tin]:
                success_ratio += 1
        final_success_ratio = success_ratio / total_num_points
        logger.debug('Classification success is %s %% \n', 100 * final_success_ratio)
        self._ret['test_success_ratio'] = final_success_ratio
        self._ret['testing_accuracy'] = final_success_ratio

        return final_success_ratio

    def predict(self, data):
        """
        predict using the svm
        Args:
            data (numpy.ndarray): the points
        Returns:
            np.ndarray: l sign
        """
        alphas = self._ret['svm']['alphas']
        bias = self._ret['svm']['bias']
        svms = self._ret['svm']['support_vectors']
        yin = self._ret['svm']['yin']
        kernel_matrix = self.construct_kernel_matrix(data, svms, self.gamma)
        self._ret['kernel_matrix_prediction'] = kernel_matrix

        total_num_points = len(data)
        lsign = np.zeros(total_num_points)
        for tin in range(total_num_points):
            ltot = 0
            for sin in range(len(svms)):
                _l = yin[sin] * alphas[sin] * kernel_matrix[tin][sin]
                ltot += _l
            lsign[tin] = np.int((np.sign(ltot + bias) + 1.) / 2.)
        self._ret['predicted_labels'] = lsign
        return lsign

    def run(self):
        """
        put the train, test, predict together
        """

        self.train(self.training_dataset[0], self.training_dataset[1])
        if self.test_dataset is not None:
            self.test(self.test_dataset[0], self.test_dataset[1])

        if self.datapoints is not None:
            predicted_labels = self.predict(self.datapoints)
            predicted_classes = map_label_to_class_name(predicted_labels, self.label_to_class)
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
            self.class_to_label = model_npz['class_to_label']
            self.label_to_class = model_npz['label_to_class']
        except KeyError as e:
            logger.warning("The model saved in Aqua 0.5 does not contain "
                           "the mapping between class names and labels. "
                           "Please setup them and save the model again "
                           "for further use. Error: %s", str(e))

    def save_model(self, file_path):
        """ save model """
        model = {'alphas': self._ret['svm']['alphas'],
                 'bias': self._ret['svm']['bias'],
                 'support_vectors': self._ret['svm']['support_vectors'],
                 'yin': self._ret['svm']['yin'],
                 'class_to_label': self.class_to_label,
                 'label_to_class': self.label_to_class}
        np.savez(file_path, **model)
