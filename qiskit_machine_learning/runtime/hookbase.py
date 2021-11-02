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

"""HookBase for the Torch Runtime"""

from typing import Optional, Any


class HookBase:
    """Base class for hooks that can be registered in ``Trainer``.
    A hook can implement 6 methods. Each method is called before/after the corresponding processes.
    """

    def __init__(self):
        self._trainer: Optional["TorchTrainer"] = None

    @property
    def trainer(self) -> Any:
        """Return the trainer for the hook"""
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: Any) -> None:
        """Set the trainer"""
        self._trainer = trainer

    def before_train(self):
        """Called before the first iteration."""
        pass

    def after_train(self):
        """Called after the last iteration."""
        pass

    def before_epoch(self):
        """Called before each epoch."""
        pass

    def after_epoch(self):
        """Called after each epoch."""
        pass

    def before_step(self):
        """Called before each iteration."""
        pass

    def after_step(self):
        """Called after each iteration."""
        pass
