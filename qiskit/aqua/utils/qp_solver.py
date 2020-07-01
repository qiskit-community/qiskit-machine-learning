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

""" qp solver """

import warnings
from typing import Optional, Tuple
import logging

import numpy as np
try:
    import cvxpy
    HAS_CVX = True
except ImportError:
    HAS_CVX = False

logger = logging.getLogger(__name__)


def optimize_svm(kernel_matrix: np.ndarray,
                 y: np.ndarray,
                 scaling: Optional[float] = None,
                 maxiter: int = 500,
                 show_progress: bool = False,
                 max_iters: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solving quadratic programming problem for SVM; thus, some constraints are fixed.

    Args:
        kernel_matrix: NxN array
        y: Nx1 array
        scaling: the scaling factor to renormalize the `y`, if it is None,
                 use L2-norm of `y` for normalization
        maxiter: number of iterations for QP solver
        show_progress: showing the progress of QP solver
        max_iters: Deprecated, use maxiter.

    Returns:
        np.ndarray: Sx1 array, where S is the number of supports
        np.ndarray: Sx1 array, where S is the number of supports
        np.ndarray: Sx1 array, where S is the number of supports

    Raises:
        NameError: If cvxpy is not installed
    """
    # pylint: disable=invalid-name, unused-argument
    if not HAS_CVX:
        raise NameError("The CVXPY package is required to use the "
                        "optimize_svm() function. You can install it with "
                        "'pip install qiskit-aqua[cvx]'.")
    if max_iters is not None:
        warnings.warn('The max_iters parameter is deprecated as of '
                      '0.8.0 and will be removed no sooner than 3 months after the release. '
                      'You should use maxiter instead.',
                      DeprecationWarning)
        maxiter = max_iters
    if y.ndim == 1:
        y = y[:, np.newaxis]
    H = np.outer(y, y) * kernel_matrix
    f = -np.ones(y.shape)
    if scaling is None:
        scaling = np.sum(np.sqrt(f * f))
    f /= scaling

    tolerance = 1e-2
    n = kernel_matrix.shape[1]

    P = np.array(H)
    q = np.array(f)
    G = -np.eye(n)
    h = np.zeros(n)
    A = y.reshape(y.T.shape)
    b = np.zeros((1, 1))
    x = cvxpy.Variable(n)
    prob = cvxpy.Problem(
        cvxpy.Minimize((1 / 2) * cvxpy.quad_form(x, P) + q.T@x),
        [G@x <= h,
         A@x == b])
    prob.solve(verbose=show_progress, qcp=True)
    result = np.asarray(x.value).reshape((n, 1))
    alpha = result * scaling
    avg_y = np.sum(y)
    avg_mat = (alpha * y).T.dot(kernel_matrix.dot(np.ones(y.shape)))
    b = (avg_y - avg_mat) / n

    support = alpha > tolerance
    logger.debug('Solving QP problem is completed.')
    return alpha.flatten(), b.flatten(), support.flatten()
