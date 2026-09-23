Qiskit Machine Learning v0.10 Migration Guide
=============================================

This guide covers the removal of deprecated qubit auto-alignment that was
announced in version ``0.9.1``. See
`#1010 <https://github.com/qiskit-community/qiskit-machine-learning/issues/1010>`__
and
`#1024 <https://github.com/qiskit-community/qiskit-machine-learning/issues/1024>`__
for background.

What changed
------------

The ``num_qubits`` argument has been removed from:

- :func:`~qiskit_machine_learning.circuit.library.qnn_circuit`
- :class:`~qiskit_machine_learning.algorithms.classifiers.VQC`
- :class:`~qiskit_machine_learning.algorithms.regressors.VQR`
- :func:`~qiskit_machine_learning.utils.derive_num_qubits_feature_map_ansatz`

The library no longer pads or resizes ``feature_map`` and ``ansatz`` circuits to
force a common qubit count. If both circuits are provided, they must already
have the same number of qubits. Quantum kernels also no longer attempt to resize
the feature map when the input feature dimension does not match.

How to migrate
--------------

Pass explicit circuits instead of ``num_qubits``:

.. code:: python

    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.algorithms.classifiers import VQC
    from qiskit_machine_learning.circuit.library import qnn_circuit

    # before
    qc, fm_params, anz_params = qnn_circuit(num_qubits=2)
    classifier = VQC(num_qubits=2)

    # after
    feature_map = zz_feature_map(2)
    qc, fm_params, anz_params = qnn_circuit(feature_map=feature_map)
    classifier = VQC(feature_map=feature_map)

When only one circuit is provided, defaults are still created at the matching
qubit count:

.. code:: python

    from qiskit.circuit.library import real_amplitudes
    from qiskit_machine_learning.circuit.library import qnn_circuit

    # uses the default ZZFeatureMap with 2 qubits
    qnn_circuit(ansatz=real_amplitudes(2))

If the feature map and ansatz have different numbers of qubits, a
``QiskitMachineLearningError`` is raised:

.. code:: python

    from qiskit.circuit.library import real_amplitudes, z_feature_map
    from qiskit_machine_learning.algorithms.classifiers import VQC

    # raises QiskitMachineLearningError
    VQC(feature_map=z_feature_map(1), ansatz=real_amplitudes(2))

For quantum kernels, construct the feature map with the expected feature
dimension before evaluation:

.. code:: python

    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    feature_map = ZZFeatureMap(3)
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    kernel.evaluate([[1.0, 2.0, 3.0]])
