# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Support Vector Classifier"""

import warnings
from typing import Optional, Type
import pickle
from pathlib import Path

from sklearn.svm import SVC

from qiskit_machine_learning.algorithms.serializable_model import SerializableModelMixin
from qiskit_machine_learning.exceptions import QiskitMachineLearningWarning
from qiskit_machine_learning.kernels import BaseKernel, FidelityQuantumKernel

from ...utils import algorithm_globals


class QSVC(SVC, SerializableModelMixin):
    r"""Quantum Support Vector Classifier that extends the scikit-learn
    `sklearn.svm.SVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_
    classifier and introduces an additional `quantum_kernel` parameter.

    This class shows how to use a quantum kernel for classification. The class inherits its methods
    like ``fit`` and ``predict`` from scikit-learn, see the example below.
    Read more in the `scikit-learn user guide
    <https://scikit-learn.org/stable/modules/svm.html#svm-classification>`_.

    **Example**

    .. code-block::

        qsvc = QSVC(quantum_kernel=qkernel)
        qsvc.fit(sample_train,label_train)
        qsvc.predict(sample_test)
    """

    def __init__(self, *, quantum_kernel: Optional[BaseKernel] = None, **kwargs):
        """
        Args:
            quantum_kernel: A quantum kernel to be used for classification.
                Has to be ``None`` when a precomputed kernel is used. If None,
                default to :class:`~qiskit_machine_learning.kernels.FidelityQuantumKernel`.
            *args: Variable length argument list to pass to SVC constructor.
            **kwargs: Arbitrary keyword arguments to pass to SVC constructor.

        Raises:
            QiskitMachineLearningWarning: If the deprecated 'kernel' kwarg is used.
        """
        if "kernel" in kwargs:
            msg = (
                "'kernel' argument is not supported and will be discarded, "
                "please use 'quantum_kernel' instead."
            )
            warnings.warn(msg, QiskitMachineLearningWarning, stacklevel=2)
            # if we don't delete, then this value clashes with our quantum kernel
            del kwargs["kernel"]

        if quantum_kernel is None:
            msg = "No quantum kernel is provided, SamplerV1 based quantum kernel will be used."
            warnings.warn(msg, QiskitMachineLearningWarning, stacklevel=2)

        self._quantum_kernel = quantum_kernel if quantum_kernel else FidelityQuantumKernel()

        if "random_state" not in kwargs:
            kwargs["random_state"] = algorithm_globals.random_seed

        super().__init__(kernel=self._quantum_kernel.evaluate, **kwargs)

    @property
    def quantum_kernel(self) -> BaseKernel:
        """Returns quantum kernel"""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: BaseKernel):
        """Sets quantum kernel"""
        self._quantum_kernel = quantum_kernel
        self.kernel = self._quantum_kernel.evaluate

    # we override this method to be able to pretty print this instance
    @classmethod
    def _get_param_names(cls) -> list[str]:
        """
        Include 'quantum_kernel' in the list of SVC parameters for cloning.

        Returns:
            list[str]: Sorted list of parameter names.
        """
        names = SVC._get_param_names()
        # Remove the base 'kernel' parameter; we override it
        if "kernel" in names:
            names.remove("kernel")
        return sorted(names + ["quantum_kernel"])

    def save(self, folder: str, filename: str = "qsvc.pkl") -> None:
        """
        Serialize the entire QSVC object to disk, including fitted parameters and
        the quantum kernel state.

        Args:
            folder (str): Directory path where the model file will be saved.
            filename (str): Name of the pickle file (default: 'qsvc.pkl').

        Raises:
            OSError: If the directory cannot be created.
            IOError: If the file cannot be written.
        """
        folder_path = Path(folder)
        try:
            folder_path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise IOError(f"Failed to create directory '{folder}': {error}") from error

        file_path = folder_path / filename
        try:
            with file_path.open("wb") as file:
                pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError as error:
            raise IOError(f"Failed to save QSVC to '{file_path}': {error}") from error

    @classmethod
    def load(
        cls: Type["QSVC"],
        folder: str,
        filename: str = "qsvc.pkl",
    ) -> "QSVC":
        """
        Deserialize a QSVC object previously saved with `save()`.

        Args:
            folder (str): Directory path where the model file is located.
            filename (str): Name of the pickle file (default: 'qsvc.pkl').

        Returns:
            QSVC: Restored QSVC instance with its quantum kernel.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            TypeError: If the loaded object is not a QSVC instance.
            OSError: For other I/O or unpickling errors.
        """
        file_path = Path(folder) / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Model file not found at '{file_path}'.")

        try:
            with file_path.open("rb") as file:
                obj = pickle.load(file)
        except OSError as error:
            raise IOError(f"Failed to load QSVC from '{file_path}': {error}") from error

        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a QSVC (got {type(obj).__name__}).")

        return obj

    def __getstate__(self):  # pragma: no cover
        """
        Define picklable state for the QSVC instance.

        Returns:
            dict: A shallow copy of the instance __dict__.
        """
        return self.__dict__.copy()

    def __setstate__(self, state):  # pragma: no cover
        """
        Restore QSVC state from unpickling.

        Args:
            state (dict): State dictionary previously returned by __getstate__.
        """
        self.__dict__.update(state)
