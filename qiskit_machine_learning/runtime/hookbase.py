# This code is part of qiskit-runtime.
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

class HookBase:
    """
    Base class for hooks that can be registered in ``Trainer``.
    Each hook can implement 6 methods. The way they are called is demonstrated
    in the following snippet:
    ```
        hook.before_train()
        for epoch in range(epochs):
            hook.before_epoch()
            for batch in train_loader:
                hook.before_step()
                trainer.run_step()
                hook.after_step()
            hook.after_epoch()
        hook.after_train()
    ```
    In the hook method, users can access ``self.trainer`` to access more
    properties about the context (e.g., model, current iteration, or config).
    """

    # Weak reference to the trainer object set by the trainer when the hook is registered.
    def __init__(self):
        self._trainer: "Trainer" = None

    @property
    def trainer(self) -> "Trainer":
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: "Trainer") -> None:
        self._trainer = trainer

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_epoch(self):
        """
        Called before each epoch.
        """
        pass

    def after_epoch(self):
        """
        Called after each epoch.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass
