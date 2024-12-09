# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Adam and AMSGRAD optimizers."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any
import os

import csv
import numpy as np
from .optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT

CALLBACK = Callable[[int, POINT, float], None]

# pylint: disable=invalid-name


class ADAM(Optimizer):
    """Adam and AMSGRAD optimizers.

    Adam [1] is a gradient-based optimization algorithm that is relies on adaptive estimates of
    lower-order moments. The algorithm requires little memory and is invariant to diagonal
    rescaling of the gradients. Furthermore, it is able to cope with non-stationary objective
    functions and noisy and/or sparse gradients.

    AMSGRAD [2] (a variant of Adam) uses a 'long-term memory' of past gradients and, thereby,
    improves convergence properties.

    References:

        [1]: Kingma, Diederik & Ba, Jimmy (2014), Adam: A Method for Stochastic Optimization.
             `arXiv:1412.6980 <https://arxiv.org/abs/1412.6980>`_

        [2]: Sashank J. Reddi and Satyen Kale and Sanjiv Kumar (2018),
             On the Convergence of Adam and Beyond.
             `arXiv:1904.09237 <https://arxiv.org/abs/1904.09237>`_
    """

    _OPTIONS = [
        "maxiter",
        "tol",
        "lr",
        "beta_1",
        "beta_2",
        "noise_factor",
        "eps",
        "amsgrad",
        "snapshot_dir",
    ]

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        maxiter: int = 10000,
        tol: float = 1e-6,
        lr: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        noise_factor: float = 1e-8,
        eps: float = 1e-10,
        amsgrad: bool = False,
        snapshot_dir: str | None = None,
        callback: CALLBACK | None = None,
    ) -> None:
        """
        Args:
            maxiter: Maximum number of iterations
            tol: Tolerance for termination
            lr: Value >= 0, Learning rate.
            beta_1: Value in range 0 to 1, Generally close to 1.
            beta_2: Value in range 0 to 1, Generally close to 1.
            noise_factor: Value >= 0, Noise factor
            eps : Value >=0, Epsilon to be used for finite differences if no analytic
                gradient method is given.
            amsgrad: True to use AMSGRAD, False if not
            snapshot_dir: If not None save the optimizer's parameter
                after every step to the given directory
            callback: A callback function passed information in each iteration step.
                The information is, in this order: current time step, the parameters, the function value.
        """
        super().__init__()
        self.callback = callback
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                self._options[k] = v
        self._maxiter = maxiter
        self._snapshot_dir = snapshot_dir
        self._tol = tol
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._noise_factor = noise_factor
        self._eps = eps
        self._amsgrad = amsgrad

        # runtime variables
        self._t = 0  # time steps
        self._m = np.zeros(1)
        self._v = np.zeros(1)
        if self._amsgrad:
            self._v_eff = np.zeros(1)

        if self._snapshot_dir is not None:
            file_path = os.path.join(self._snapshot_dir, "adam_params.csv")
            if not os.path.isfile(file_path):
                # pylint: disable=unspecified-encoding
                with open(file_path, mode="w") as csv_file:
                    fieldnames = ["v", "v_eff", "m", "t"] if self._amsgrad else ["v", "m", "t"]
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()

    @property
    def settings(self) -> dict[str, Any]:
        return {
            "maxiter": self._maxiter,
            "tol": self._tol,
            "lr": self._lr,
            "beta_1": self._beta_1,
            "beta_2": self._beta_2,
            "noise_factor": self._noise_factor,
            "eps": self._eps,
            "amsgrad": self._amsgrad,
            "snapshot_dir": self._snapshot_dir,
        }

    def get_support_level(self):
        """Return support level dictionary"""
        return {
            "gradient": OptimizerSupportLevel.supported,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.supported,
        }

    def save_params(self, snapshot_dir: str) -> None:
        """Save the current iteration parameters to a file called ``adam_params.csv``.

        Note:

            The current parameters are appended to the file, if it exists already.
            The file is not overwritten.

        Args:
            snapshot_dir: The directory to store the file in.
        """
        # pylint: disable=unspecified-encoding
        file_path = os.path.join(snapshot_dir, "adam_params.csv")

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        fieldnames = ["v", "v_eff", "m", "t"] if self._amsgrad else ["v", "m", "t"]

        with open(file_path, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            row = {"v": self._v, "m": self._m, "t": self._t}
            if self._amsgrad:
                row["v_eff"] = self._v_eff
            writer.writerow(row)

    def load_params(self, load_dir: str) -> None:
        """Load iteration parameters for a file called ``adam_params.csv``.

        Args:
            load_dir: The directory containing ``adam_params.csv``.
        """
        # pylint: disable=unspecified-encoding
        file_path = os.path.join(load_dir, "adam_params.csv")

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with open(file_path, mode="r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)

            for line in reader:
                self._v = np.fromstring(line["v"].strip("[]"), dtype=float, sep=" ")
                if self._amsgrad:
                    self._v_eff = np.fromstring(line["v_eff"].strip("[]"), dtype=float, sep=" ")
                self._m = np.fromstring(line["m"].strip("[]"), dtype=float, sep=" ")
                self._t = int(line["t"].strip("[]"))

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        """Minimize the scalar function.

        Args:
            fun: The scalar function to minimize.
            x0: The initial point for the minimization.
            jac: The gradient of the scalar function ``fun``.
            bounds: Bounds for the variables of ``fun``. This argument might be ignored if the
                optimizer does not support bounds.
        Returns:
            The result of the optimization, containing e.g. the result as attribute ``x``.
        """
        if jac is None:
            jac = Optimizer.wrap_function(Optimizer.gradient_num_diff, (fun, self._eps))

        derivative = jac(x0)
        self._t = 0
        self._m = np.zeros(np.shape(derivative))
        self._v = np.zeros(np.shape(derivative))
        if self._amsgrad:
            self._v_eff = np.zeros(np.shape(derivative))

        params = params_new = x0
        while self._t < self._maxiter:
            if self._t > 0:
                derivative = jac(params)
            self._t += 1
            self._m = self._beta_1 * self._m + (1 - self._beta_1) * derivative
            self._v = self._beta_2 * self._v + (1 - self._beta_2) * derivative * derivative
            lr_eff = self._lr * np.sqrt(1 - self._beta_2**self._t) / (1 - self._beta_1**self._t)
            if not self._amsgrad:
                params_new = params - lr_eff * self._m.flatten() / (
                    np.sqrt(self._v.flatten()) + self._noise_factor
                )
            else:
                self._v_eff = np.maximum(self._v_eff, self._v)
                params_new = params - lr_eff * self._m.flatten() / (
                    np.sqrt(self._v_eff.flatten()) + self._noise_factor
                )

            if self._snapshot_dir is not None:
                self.save_params(self._snapshot_dir)

            if self.callback is not None:
                self.callback(self._t, params_new, fun(params_new))

            # check termination
            if np.linalg.norm(params - params_new) < self._tol:
                break

            params = params_new

        result = OptimizerResult()
        result.x = params_new
        result.fun = fun(params_new)
        result.nfev = self._t
        return result
