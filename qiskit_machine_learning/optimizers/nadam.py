# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""NAdam Optimizer"""

from __future__ import annotations
import os
import numpy as np
from typing import Callable
from .optimizer import Optimizer, OptimizerResult, POINT, OptimizerSupportLevel


class NAdam(Optimizer):
    """NAdam optimizer (Nesterov-accelerated Adaptive Moment Estimation)."""

    def __init__(
        self,
        maxiter: int = 200,
        tol: float = 1e-6,
        lr: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        noise_factor: float = 1e-8,
        callback: Callable | None = None,
        snapshot_dir: str | None = None,
    ) -> None:
        """Initialize NAdam optimizer."""
        super().__init__()
        self.maxiter = maxiter
        self.tol = tol
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.noise_factor = noise_factor
        self.callback = callback
        self.snapshot_dir = snapshot_dir

        # Internal state
        self._m = None
        self._v = None
        self._t = 0

    def get_support_level(self):
        """Return the support level for NAdam optimizer."""
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }

    @property
    def settings(self):
        """Return optimizer settings as a dictionary."""
        return {
            "maxiter": self.maxiter,
            "tol": self.tol,
            "lr": self.lr,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "eps": self.eps,
            "noise_factor": self.noise_factor,
            "callback": self.callback,
            "snapshot_dir": self.snapshot_dir,
        }

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        """Minimize the scalar function using NAdam."""

        result = OptimizerResult()
        x = np.array(x0, dtype=float)
        self._m = np.zeros_like(x)
        self._v = np.zeros_like(x)
        self._t = 0

        for i in range(self.maxiter):
            self._t += 1

            # Compute gradient numerically
            grad = self.gradient_num_diff(x, fun, self.eps)

            # Add optional stochastic noise
            grad += self.noise_factor * np.random.randn(*grad.shape)

            # NAdam update rule
            m_hat = self.beta_1 * self._m + (1 - self.beta_1) * grad
            v_hat = self.beta_2 * self._v + (1 - self.beta_2) * (grad ** 2)

            m_corr = m_hat / (1 - self.beta_1 ** self._t)
            v_corr = v_hat / (1 - self.beta_2 ** self._t)

            # Nesterov momentum
            x_update = self.lr * (self.beta_1 * m_corr + (1 - self.beta_1) * grad / (1 - self.beta_1 ** self._t)) / (np.sqrt(v_corr) + self.eps)
            x -= x_update

            # Update state
            self._m = m_hat
            self._v = v_hat

            fval = fun(x)

            # Callback
            if self.callback is not None:
                self.callback(self._t, x, fval)

            # Save snapshot
            if self.snapshot_dir is not None:
                np.save(os.path.join(self.snapshot_dir, f"nadam_m_{i}.npy"), self._m)
                np.save(os.path.join(self.snapshot_dir, f"nadam_v_{i}.npy"), self._v)
                np.save(os.path.join(self.snapshot_dir, f"nadam_x_{i}.npy"), x)

            # Check convergence
            if np.linalg.norm(x_update) < self.tol:
                break

        result.x = x
        result.fun = fun(x)
        result.nfev = self._t
        result.nit = self._t

        return result

    def load_params(self, snapshot_dir: str):
        """Load optimizer state from snapshot files."""
        last_iter = max([int(f.split("_")[-1].split(".")[0])
                        for f in os.listdir(snapshot_dir) if f.startswith("nadam_x_")], default=-1)
        if last_iter >= 0:
            self._m = np.load(os.path.join(snapshot_dir, f"nadam_m_{last_iter}.npy"))
            self._v = np.load(os.path.join(snapshot_dir, f"nadam_v_{last_iter}.npy"))
            # x can also be restored if needed
            # x = np.load(os.path.join(snapshot_dir, f"nadam_x_{last_iter}.npy"))
            self._t = last_iter + 1
