# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Support Vector Classifier"""

from typing import Optional, Union, Sequence

import numpy as np
from sklearn.svm import SVC

from qiskit import Aer
from qiskit.utils.algorithm_globals import algorithm_globals
from qiskit_machine_learning.kernels.quantum_kernel import QuantumKernel
from qiskit_machine_learning.algorithms.kernel_trainers import QuantumKernelTrainer


class QSVC(SVC):
    r"""Quantum Support Vector Classifier.

    This class shows how to use a quantum kernel for classification. The class extends
    `sklearn.svm.SVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_,
    and thus inherits its methods like ``fit`` and ``predict`` used in the example below.
    Read more in the `sklearn user guide
    <https://scikit-learn.org/stable/modules/svm.html#svm-classification>`_.

    **Example**

    .. code-block::python
        qsvc = QSVC(quantum_kernel=quant_kernel)
        qsvc.fit(sample_train,label_train)
        qsvc.predict(sample_test)
    """

    def __init__(
        self,
        quantum_kernel: Optional[QuantumKernel] = None,
        kernel_trainer: Optional[QuantumKernelTrainer] = None,
        regularization: Optional[float] = 1.0,
        degree: Optional[int] = 3,
        gamma: Optional[Union[str, float]] = "scale",
        coef0: Optional[float] = 0.0,
        shrinking: Optional[bool] = True,
        probability: Optional[bool] = False,
        tol: Optional[float] = 1e-3,
        cache_size: Optional[float] = 200,
        class_weight: Optional[Union[dict, str]] = None,
        verbose: Optional[bool] = False,
        max_iter: Optional[int] = -1,
        decision_function_shape: Optional[str] = "ovr",
        break_ties: Optional[bool] = False,
        random_state: Optional[int] = algorithm_globals.random_seed,
    ):
        r"""
        Args:
            regularization: Regularization parameter. The strength of the regularization is
                        inversely proportional to regularization. Must be strictly positive. The penalty
                        is a squared l2 penalty.
            quantum_kernel: QuantumKernel to be used for classification.
            kernel_trainer: QuantumKernelTrainer to be used for kernel optimization.
            degree: Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
            gamma {‘scale’, ‘auto’} or float: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
                        * if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as
                        value of gamma,
                        * if ‘auto’, uses 1 / n_features.
            coef0: Independent term in kernel function. It is only significant in ‘poly’ and
                        ‘sigmoid’.
            shrinking: Whether to use the shrinking heuristic.
            probability: Whether to enable probability estimates. This must be enabled prior to
                        calling fit, will slow down that method as it internally uses 5-fold
                        cross-validation, and `predict_proba` may be inconsistent with `predict`.
            tol: Tolerance for stopping criterion.
            cache_size: Specify the size of the kernel cache (in MB).
            class_weight (dict or 'balanced'): Set the parameter C of class i to class_weight[i]*C
                        for SVC. If not given, all classes are supposed to have weight one. The
                        “balanced” mode uses the values of y to automatically adjust weights inversely
                        proportional to class frequencies in the input data as
                        n_samples / (n_classes * np.bincount(y)).
            verbose: Enable verbose output. Note that this setting takes advantage of a per-process
                        runtime setting in libsvm that, if enabled, may not work properly in a
                        multithreaded context.
            max_iter: Hard limit on iterations within solver, or -1 for no limit.
            decision_function_shape {'ovo', 'ovr'}: Whether to return a one-vs-rest (‘ovr’) decision
                        function of shape (n_samples, n_classes) as all other classifiers, or the
                        original one-vs-one (‘ovo’) decision function of libsvm which has shape
                        (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one (‘ovo’) is
                        always used as multi-class strategy. The parameter is ignored for binary
                        classification.
            break_ties: If true, decision_function_shape='ovr', and number of classes > 2, predict will
                        break ties according to the confidence values of decision_function; otherwise
                        the first class among the tied classes is returned. Please note that breaking
                        ties comes at a relatively high computational cost compared to a simple predict.
            random_state: Controls the pseudo random number generation for shuffling the data for
                        probability estimates. Ignored when probability is False. Pass an int for
                        reproducible output across multiple function calls.
        """

        # Class fields
        self._quantum_kernel = None
        self._kernel_trainer = None

        # Setters
        self.quantum_kernel = (
            quantum_kernel
            if quantum_kernel
            else QuantumKernel(quantum_instance=Aer.get_backend("statevector_simulator"))
        )
        self.kernel_trainer = kernel_trainer

        super().__init__(
            C=regularization,
            kernel=self.quantum_kernel.evaluate,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )

    @property
    def quantum_kernel(self) -> QuantumKernel:
        """Returns quantum kernel"""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: QuantumKernel):
        """Sets quantum kernel"""
        # QuantumKernel passed
        self._quantum_kernel = quantum_kernel

    @property
    def kernel_trainer(self) -> QuantumKernelTrainer:
        """Returns quantum kernel trainer"""
        return self._kernel_trainer

    @kernel_trainer.setter
    def kernel_trainer(self, qk_trainer: QuantumKernelTrainer) -> None:
        """Returns quantum kernel trainer"""
        self._kernel_trainer = qk_trainer

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[Sequence[float]] = None
    ) -> SVC:
        r"""
        Wrapper method for SVC.fit which optimizes the quantum kernel's
        user parameters before fitting the SVC.

        Args:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)
                Training vector, where `n_samples` is the number of samples and
                `n_features` is the number of features.
            y: array-like of shape (n_samples,)
                Target vector relative to X.
            sample_weight : array-like of shape (n_samples,), default=None
                Array of weights that are assigned to individual
                samples. If not provided,
                then each sample is given unit weight.

        Returns:
            SVC: Returns instance of the trained classifier

        Raises:
            ValueError: Unbound user parameters on feature map
        """

        unbound_user_params = self._quantum_kernel.unbound_user_parameters()
        if (len(unbound_user_params) > 0) and (self._kernel_trainer is None):
            raise ValueError(
                f"""
            Cannot fit QSVC while feature map has unbound user parameters ({unbound_user_params}).
            """
            )

        # Conduct kernel optimization, if required
        if self._kernel_trainer:
            results = self._kernel_trainer.fit_kernel(self.quantum_kernel, X, y)
            self.quantum_kernel.assign_user_parameters(results.optimal_parameters)

        return super().fit(X=X, y=y, sample_weight=sample_weight)
