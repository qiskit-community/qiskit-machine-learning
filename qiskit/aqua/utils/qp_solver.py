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

import logging

import numpy as np
from cvxopt import matrix, solvers

logger = logging.getLogger(__name__)


def optimize_svm(kernel_matrix, y, scaling=None, max_iters=500, show_progress=False):
    """
    Solving quadratic programming problem for SVM; thus, some constraints are fixed.

    The notation is follows the equation here:
    http://cvxopt.org/userguide/coneprog.html#quadratic-programming

    Args:
        kernel_matrix (numpy.ndarray): NxN array
        y (numpy.ndarray): Nx1 array
        scaling (float): the scaling factor to renormalize the `y`, if it is None,
                            use L2-norm of `y` for normalization
        max_iters (int): number of iterations for QP solver
        show_progress (bool): showing the progress of QP solver

    Returns:
        numpy.ndarray: Sx1 array, where S is the number of supports
        numpy.ndarray: Sx1 array, where S is the number of supports
        numpy.ndarray: Sx1 array, where S is the number of supports
    """
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
