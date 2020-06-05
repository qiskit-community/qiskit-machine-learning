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
The Sklearn SVM algorithm (classical).
"""

from typing import Dict, Optional, Union
import logging
import warnings
import numpy as np
from qiskit.aqua import AquaError
from qiskit.aqua.algorithms import ClassicalAlgorithm
from qiskit.aqua.utils import get_num_classes
from qiskit.aqua.components.multiclass_extensions import MulticlassExtension
from ._sklearn_svm_binary import _SklearnSVMBinary
from ._sklearn_svm_multiclass import _SklearnSVMMulticlass
from ._rbf_svc_estimator import _RBF_SVC_Estimator

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name


class SklearnSVM(ClassicalAlgorithm):
    """
    The Sklearn SVM algorithm (classical).

    This scikit-learn based SVM algorithm uses a classical approach to experiment with feature map
    classification problems. See also the quantum classifier :class:`QSVM`.

    Internally, this algorithm will run the binary classification or multiclass classification
    based on how many classes the data has. If the data has more than 2 classes then a
    *multiclass_extension* is required to be supplied. Aqua provides several
    :mod:`~qiskit.aqua.components.multiclass_extensions`.
    """

    def __init__(self, training_dataset: Dict[str, np.ndarray],
                 test_dataset: Optional[Dict[str, np.ndarray]] = None,
                 datapoints: Optional[np.ndarray] = None,
                 gamma: Optional[int] = None,
                 multiclass_extension: Optional[MulticlassExtension] = None) -> None:
        # pylint: disable=line-too-long
        """
        Args:
            training_dataset: Training dataset.
            test_dataset: Testing dataset.
            datapoints: Prediction dataset.
            gamma: Used as input for sklearn rbf_kernel which is used internally. See
                `sklearn.metrics.pairwise.rbf_kernel
                <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html>`_
                for more information about gamma.
            multiclass_extension: If number of classes is greater than 2 then a multiclass scheme
                must be supplied, in the form of a multiclass extension.

        Raises:
            AquaError: Multiclass extension not supplied when number of classes > 2
        """
        super().__init__()
        if training_dataset is None:
            raise AquaError('Training dataset must be provided.')

        is_multiclass = get_num_classes(training_dataset) > 2
        if is_multiclass:
            if multiclass_extension is None:
                raise AquaError('Dataset has more than two classes. '
                                'A multiclass extension must be provided.')
        else:
            if multiclass_extension is not None:
                logger.warning("Dataset has just two classes. Supplied multiclass "
                               "extension will be ignored")

        svm_instance = None  # type: Optional[Union[_SklearnSVMBinary, _SklearnSVMMulticlass]]
        if multiclass_extension is None:
            svm_instance = _SklearnSVMBinary(training_dataset, test_dataset, datapoints, gamma)
        else:
            multiclass_extension.set_estimator(_RBF_SVC_Estimator, [])
            svm_instance = _SklearnSVMMulticlass(
                training_dataset, test_dataset, datapoints, gamma, multiclass_extension)

        self.instance = svm_instance

    def train(self, data, labels):
        """
        Train the SVM

        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                D is the feature dimension.
            labels (numpy.ndarray): Nx1 array, where N is the number of data
        """
        self.instance.train(data, labels)

    def test(self, data, labels):
        """
        Test the SVM

        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                D is the feature dimension.
            labels (numpy.ndarray): Nx1 array, where N is the number of data

        Returns:
            float: accuracy
        """
        return self.instance.test(data, labels)

    def predict(self, data):
        """
        Predict using the SVM

        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
        Returns:
            numpy.ndarray: predicted labels, Nx1 array
        """
        return self.instance.predict(data)

    def _run(self):
        return self.instance.run()

    @property
    def label_to_class(self):
        """ returns label to class """
        return self.instance.label_to_class

    @property
    def class_to_label(self):
        """ returns class to label """
        return self.instance.class_to_label

    @property
    def ret(self):
        """ returns result """
        return self.instance.ret

    @ret.setter
    def ret(self, new_ret):
        """ sets result """
        self.instance.ret = new_ret

    def load_model(self, file_path):
        """
        Load a model from a file path.

        Args:
            file_path (str): the path of the saved model.
        """
        self.instance.load_model(file_path)

    def save_model(self, file_path):
        """
        Save the model to a file path.

        Args:
            file_path (str): a path to save the model.
        """
        self.instance.save_model(file_path)


class SVM_Classical(SklearnSVM):
    """ The deprecated Sklearn SVM algorithm. """

    def __init__(self, training_dataset: Dict[str, np.ndarray],
                 test_dataset: Optional[Dict[str, np.ndarray]] = None,
                 datapoints: Optional[np.ndarray] = None,
                 gamma: Optional[int] = None,
                 multiclass_extension: Optional[MulticlassExtension] = None) -> None:
        warnings.warn('Deprecated class {}, use {}.'.format('SVM_Classical', 'SklearnSVM'),
                      DeprecationWarning)
        super().__init__(training_dataset,
                         test_dataset,
                         datapoints,
                         gamma,
                         multiclass_extension)
