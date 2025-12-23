.. _qiskit-machine-learning-primitives:

.. automodule:: qiskit_machine_learning.primitives
    :no-members:
    :no-inherited-members:
    :no-special-members:


Qiskit Machine Learning (here QML for short) are thin wrappers around Qiskit's
Statevector primitives (V2). In addition to these, they provide an *exact* mode of
execution, which computes the full statevector, rather than randomly sampling it. This
mimics the behavior of early V1 statevector primitives, providing a familiar
Qiskit *V2 primitives* interface (PUB-based execution) for reliable and reproducible prototyping
of small problem instances.

QML primitives are built to satisfy the following needs:

* Provide a stable, QML-centric API for internal components (QNNs, kernels, gradients);
* Offer a fully deterministic **exact** simulation for unit tests and reference results;
* Optionally emulate **shot noise** in local simulation by delegating to Qiskit's statevector
  primitives.

Why introduce these wrappers?
-----------------------------

Qiskit's V2 primitives ecosystem standardizes execution around Primitive Unified Blocs (PUBs) and
structured result objects. This infrastructure does not provide a direct way to compute the full
statevector result without shot noise, which some unit tests and prototyping tasks benefit from.
QML primitives address this needs by allowing the exact simulation mode directly, or acting as a
light wrapper around Qiskit Statevector (V2) primitives when sampling with shot noise.

Execution modes
---------------

Exact mode (deterministic)
^^^^^^^^^^^^^^^^^^^^^^^^^^

In *exact mode*, QML computes results analytically from a statevector representation.
This is primarily intended for tests and reference calculations:

* Deterministic outputs (no sampling stochasticity);
* Fast for small circuits, but the cost scales exponentially with the number of qubits;
* Ideal for verifying gradients / batching logic.

Statevector primitive fallback (optional shot noise)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When exact mode is disabled (or not applicable), QML delegates to Qiskit's Statevector primitives.
This supports local statevector simulation and can emulate shot noise when shots are requested.


Examples of exact simulation and with shot noise
------------------------------------------------

.. code-block:: python

    # QMLEstimator example
    from qiskit.circuit import QuantumCircuit, Parameter
    from qiskit_machine_learning.primitives import QMLEstimator

    require_exact: bool = True

    # Build circuit + observable...
    # For exact methods, circuits should be small (10-20 qubits or less) because of the
    # exponential computational cost.

    if require_exact:
        # Setting precision to 0 triggers the `exact` mode, performing the calculation with
        # the estimator routines implemented in Qiskit Machine Learning
        est = QMLEstimator(default_precision=0)
    else:
        # Setting precision to a float greater than 0 triggers the `shot-noise` mode, invoking
        # StatevectorEstimator directly from `qiskit.primitives`.
        est = QMLEstimator(default_precision=0.1)

    result = est.run([(qc, [obs], [theta_values])]).result()

    # ======================================================

    # QMLSampler example
    from qiskit_machine_learning.primitives import QMLSampler

    if require_exact:
        # Setting `shots=None` triggers the `exact` mode, performing the calculation with
        # the sampler routines implemented in Qiskit Machine Learning
        sampler = QMLSampler(shots=None)
    else:
        # Setting a finite integer number of shots, instead, invokes StatevectorSampler from Qiskit
        sampler = QMLSampler(shots=10_000)

    result = sampler.run([(qc, [param_values])]).result()


Executing workloads on IBM quantum hardware
-------------------------------------------

QML primitives do not support hardware execution, as they are intended for classical simulation
of small circuit prototypes. To run workloads on IBM quantum computers, use the primitives
provided in the ``qiskit-ibm-runtime`` library.

Here, we provide an example usage of kernel alignment (based on the
`08_quantum_kernel_trainer.ipynb <docs/tutorials/08_quantum_kernel_trainer.ipynb>`_ tutorial)
using the Qiskit IBM Runtime primitives.


.. code-block:: bash

    pip install qiskit-ibm-runtime

.. code-block:: python

    import numpy as np
    from sklearn import metrics

    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.optimizers import SPSA
    from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel
    from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.datasets import ad_hoc_data

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler

    # 1. Dataset (same as tutorial)
    adhoc_dimension = 2
    X_train, y_train, X_test, y_test, _ = ad_hoc_data(
        training_size=12,
        test_size=6,
        n=adhoc_dimension,
        gap=0.3,
        plot_data=False,
        one_hot=False,
        include_sample_total=True,
    )

    # 2. Trainable feature map (same idea)
    training_params = ParameterVector("θ", 1)
    fm0 = QuantumCircuit(2)
    fm0.ry(training_params[0], 0)
    fm0.ry(training_params[0], 1)
    fm1 = zz_feature_map(2)
    feature_map = fm0.compose(fm1)

    # 3. Choose least-busy QPU + open a Runtime Session
    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)

    # Create a transpiler pass manager targeting the backend ISA
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

    with Session(service=service, backend=backend) as session:
        sampler = Sampler(mode=session)  # V2 primitive in session mode

        # Compute-uncompute fidelity uses the Sampler primitive
        fidelity = ComputeUncompute(sampler=sampler, shots=1024, transpiler=pm)

        # Trainable fidelity quantum kernel + trainer
        qkernel = TrainableFidelityQuantumKernel(
            fidelity=fidelity,
            feature_map=feature_map,
            training_parameters=training_params,
        )

        optimizer = SPSA(maxiter=6, learning_rate=0.05, perturbation=0.05)

        qkt = QuantumKernelTrainer(
            quantum_kernel=qkernel,
            loss="svc_loss",
            optimizer=optimizer,
            initial_point=[np.pi / 2],
        )

        # 4. Kernel alignment (training)
        qka_result = qkt.fit(X_train, y_train)
        trained_kernel = qka_result.quantum_kernel

    # 5. Fit a model using the kernel
    qsvc = QSVC(quantum_kernel=trained_kernel)
    qsvc.fit(X_train, y_train)
    y_pred = qsvc.predict(X_test)

    print("Backend:", backend.name)
    print("Optimal kernel params:", qka_result.optimal_parameters)
    print("Balanced accuracy:", metrics.balanced_accuracy_score(y_true=y_test, y_pred=y_pred))

