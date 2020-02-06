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

from typing import Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

_HAS_CVXOPT = False
try:
    from cvxopt import matrix, solvers
    _HAS_CVXOPT = True
except ImportError:
    logger.info('CVXOPT is not installed. See http://cvxopt.org/install/index.html')


def optimize_svm(kernel_matrix: np.ndarray,
                 y: np.ndarray,
                 scaling: Optional[float] = None,
                 max_iters: int = 500,
                 show_progress: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solving quadratic programming problem for SVM; thus, some constraints are fixed.

    The notation is follows the equation here:
    http://cvxopt.org/userguide/coneprog.html#quadratic-programming

    Args:
        kernel_matrix: NxN array
        y: Nx1 array
        scaling: the scaling factor to renormalize the `y`, if it is None,
                 use L2-norm of `y` for normalization
        max_iters: number of iterations for QP solver
        show_progress: showing the progress of QP solver

    Returns:
        np.ndarray: Sx1 array, where S is the number of supports
        np.ndarray: Sx1 array, where S is the number of supports
        np.ndarray: Sx1 array, where S is the number of supports

    Raises:
        NameError: CVXOPT not installed.
    """
    # pylint: disable=invalid-name
    if not _HAS_CVXOPT:
        raise NameError('CVXOPT is not installed. See http://cvxopt.org/install/index.html')
    if y.ndim == 1:
        y = y[:, np.newaxis]
    H = np.outer(y, y) * kernel_matrix
    f = -np.ones(y.shape)
    if scaling is None:
        scaling = np.sum(np.sqrt(f * f))
    f /= scaling

    tolerance = 1e-2
    n = kernel_matrix.shape[1]

    P = matrix(H)
    q = matrix(f)
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    A = matrix(y, y.T.shape)
    b = matrix(np.zeros(1), (1, 1))
    solvers.options['maxiters'] = max_iters
    solvers.options['show_progress'] = show_progress

    ret = solvers.qp(P, q, G, h, A, b, kktsolver='ldl')
    alpha = np.asarray(ret['x']) * scaling
    avg_y = np.sum(y)
    avg_mat = (alpha * y).T.dot(kernel_matrix.dot(np.ones(y.shape)))
    b = (avg_y - avg_mat) / n

    support = alpha > tolerance
    logger.debug('Solving QP problem is completed.')
    return alpha.flatten(), b.flatten(), support.flatten()
