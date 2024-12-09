Qiskit Machine Learning v0.8 Migration Guide
============================================

This tutorial will guide you through the process of migrating your code
using V2 primitives.

Introduction
------------

The Qiskit Machine Learning 0.8 release focuses on transitioning from V1 to V2 primitives. 
This release also incorporates selected algorithms from the now deprecated `qiskit_algorithms` repository.


Contents:

-  Overview of the primitives
-  Transpilation and Pass Managers
-  Algorithms from `qiskit_algorithms`
-  🔪 The Sharp Bits: Common Pitfalls

Overview of the primitives
--------------------------

With the launch of `Qiskit 1.0`, V1 primitives are deprecated and replaced by V2 primitives. Further details
are available in the 
`V2 primitives migration guide <https://docs.quantum.ibm.com/migration-guides/v2-primitives>`__.

The Qiskit Machine Learning 0.8 update aligns with the Qiskit IBM Runtime’s Primitive Unified Block (PUB) 
requirements and the constraints of the instruction set architecture (ISA) for circuits and observables. 

Users can switch between `V1` primitives and `V2` primitives from version `0.8`.

**Warning**:  V1 primitives are deprecated and will be removed in version `0.9`. To ensure full compatibility 
with V2 primitives, review the transpilation and pass managers section if your primitives require transpilation, 
such as those from `qiskit-ibm-runtime`.

Usage of V2 primitives is as straightforward as using V1:

- For kernel based methods:

.. code:: ipython3

    from qiskit.primitives import StatevectorSampler as Sampler
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    ...
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler=sampler)
    feature_map = ZZFeatureMap(num_qubits)
    qk = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    ...

- For Estimator based neural_network based methods:

.. code:: ipython3

    from qiskit.primitives import StatevectorEstimator as Estimator
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
    ...
    estimator = Estimator()
    estimator_gradient = ParamShiftEstimatorGradient(estimator=estimator)
    
    estimator_qnn = EstimatorQNN(
        circuit=circuit,
        observables=observables,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
        gradient=estimator_gradient,
    )
    ...

- For Sampler based neural_network based methods:

.. code:: ipython3
    
    from qiskit.primitives import StatevectorSampler as Sampler
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit_machine_learning.gradients import ParamShiftSamplerGradient
    ...
    sampler = Sampler()
    sampler_gradient = ParamShiftSamplerGradient(sampler=sampler)

    sampler_qnn = SamplerQNN(
        circuit=circuit,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        interpret=parity,
        output_shape=output_shape,
        sampler=sampler,
        gradient=sampler_gradient,
    )
    ...


Transpilation and Pass Managers
-------------------------------
 
If your primitives require transpiled circuits,i.e. `qiskit-ibm-runtime.primitives`,
use `pass_manager` with `qiskit-machine-learning` functions to optimize performance.

- For kernel based methods:

.. code:: ipython3

    from qiskit_ibm_runtime import Session, SamplerV2
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    from qiskit_machine_learning.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    ...
    backend = GenericBackendV2(num_qubits=num_qubits)
    session = Session(backend=backend)
    pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend)

    sampler = SamplerV2(mode=session)
    fidelity = ComputeUncompute(sampler=sampler, pass_manager=pass_manager)

    feature_map = ZZFeatureMap(num_qubits)
    qk = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    ...

- For Estimator based neural_network based methods:

.. code:: ipython3

    from qiskit_ibm_runtime import Session, EstimatorV2
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient

    ...
    backend = GenericBackendV2(num_qubits=num_qubits)
    session = Session(backend=backend)

    estimator = Estimator(mode=session)
    pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend)
    estimator_qnn = EstimatorQNN(
        circuit=qc,
        observables=[observables],
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
        pass_manager=pass_manager,
    )

or with more details:

.. code:: ipython3

    backend = GenericBackendV2(num_qubits=num_qubits)
    session = Session(backend=backend)

    estimator = Estimator(mode=session)
    pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend)
    estimator_gradient = ParamShiftEstimatorGradient(
        estimator=estimator, pass_manager=pass_manager
    )

    isa_qc = pass_manager.run(qc)
    observables = SparsePauliOp.from_list(...)
    isa_observables = observables.apply_layout(isa_qc.layout)
    estimator_qnn = EstimatorQNN(
        circuit=isa_qc,
        observables=[isa_observables],
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
        gradient=estimator_gradient,
    )

- For Sampler based neural_network based methods:

.. code:: ipython3
    
    from qiskit_ibm_runtime import Session, SamplerV2
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit_machine_learning.gradients import ParamShiftSamplerGradient

    ...
    backend = GenericBackendV2(num_qubits=num_qubits)
    session = Session(backend=backend)
    pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend)
    sampler = SamplerV2(mode=session)

    sampler_qnn = SamplerQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        interpret=parity,
        output_shape=output_shape,
        sampler=sampler,
        pass_manager=pass_manager,
    )

or with more details:

.. code:: ipython3

    backend = GenericBackendV2(num_qubits=num_qubits)
    session = Session(backend=backend)
    pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend)

    sampler = SamplerV2(mode=session)
    sampler_gradient = ParamShiftSamplerGradient(sampler=sampler, pass_manager=self.pass_manager)
    isa_qc = pass_manager.run(qc)
    sampler_qnn = SamplerQNN(
        circuit=isa_qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        interpret=parity,
        output_shape=output_shape,
        sampler=sampler,
        gradient=sampler_gradient,
    )
    ...


Algorithms from `qiskit_algorithms`
-----------------------------------

Essential features of Qiskit Algorithms have been integrated into Qiskit Machine Learning.
Therefore, Qiskit Machine Learning will no longer depend on Qiskit Algorithms.
This migration requires Qiskit 1.0 or higher and may necessitate updating Qiskit Aer. 
Be cautious during updates to avoid breaking changes in critical production stages. 

Users must update their imports and code references in code that uses Qiskit Machine Leaning and Algorithms:

- Change `qiskit_algorithms.gradients` to `qiskit_machine_learning.gradients`
- Change `qiskit_algorithms.optimizers` to `qiskit_machine_learning.optimizers`
- Change `qiskit_algorithms.state_fidelities` to `qiskit_machine_learning.state_fidelities`
- Update utilities as needed due to partial merge.

To continue using sub-modules and functionalities of Qiskit Algorithms that **have not been transferred**, 
you may continue using them as before by importing from Qiskit Algorithms. However, be aware that Qiskit Algorithms
is no longer officially supported and some of its functionalities may not work in your use case. For any problems 
directly related to Qiskit Algorithms, please open a GitHub issue at 
`qiskit-algorithms <https://github.com/qiskit-community/qiskit-algorithms>`__.
Should you want to include a Qiskit Algorithms functionality that has not been incorporated in Qiskit Machine
Learning, please open a feature-request issue at 
`qiskit-machine-learning <https://github.com/qiskit-community/qiskit-machine-learning>`__,

explaining why this change would be useful for you and other users.

Four examples of upgrading the code can be found below.
  
Gradients:

.. code:: ipython3

    # Before:
    from qiskit_algorithms.gradients import SPSA, ParameterShift
    # After:
    from qiskit_machine_learning.gradients import SPSA, ParameterShift
    # Usage
    spsa = SPSA()
    param_shift = ParameterShift()

Optimizers:

.. code:: ipython3

    # Before:
    from qiskit_algorithms.optimizers import COBYLA, ADAM
    # After:
    from qiskit_machine_learning.optimizers import COBYLA, ADAM
    # Usage
    cobyla = COBYLA()
    adam = ADAM()

Quantum state fidelities:

.. code:: ipython3

    # Before:
    from qiskit_algorithms.state_fidelities import ComputeFidelity
    # After:
    from qiskit_machine_learning.state_fidelities import ComputeFidelity
    # Usage
    fidelity = ComputeFidelity()


Algorithm globals (used to fix the random seed):

.. code:: ipython3

    # Before:
    from qiskit_algorithms.utils import algorithm_globals
    # After:
    from qiskit_machine_learning.utils import algorithm_globals
    algorithm_globals.random_seed = 1234


🔪 The Sharp Bits: Common Pitfalls
-----------------------------------

- 🔪 Transpiling without measurements:

.. code:: ipython3

    # Before:
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)
    pass_manager.run(qc)

This approach causes issues for the transpiler, as it will measure all physical qubits instead 
of virtual qubits when the number of physical qubits exceeds the number of virtual qubits. 
Always add measurements before transpilation:


.. code:: ipython3

    # After:
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)
    qc.measure_all()
    pass_manager.run(qc)

- 🔪 Dynamic Attribute Naming in Qiskit v1.x:

In the latest version of Qiskit (v1.x), the dynamic naming of attributes based on the
classical register's name introduces potential bugs.
Please use `meas` or `c` for your register names to avoid any issues for SamplerV2.

.. code:: ipython3

    # for measue_all():
    dist = result[0].data.meas.get_counts()

.. code:: ipython3

    # for cbit:
    dist = result[0].data.c.get_counts()

- 🔪 Adapting observables for transpiled circuits:

.. code:: ipython3

    # Wrong:
    ...
    pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend)
    isa_qc = pass_manager.run(qc)
    observables = SparsePauliOp.from_list(...)
    estimator_qnn = EstimatorQNN(
        circuit=isa_qc,
        observables=[observables],
    ...


    # Correct:
        ...
        pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend)
        isa_qc = pass_manager.run(qc)
        observables = SparsePauliOp.from_list(...)
        isa_observables = observables.apply_layout(isa_qc.layout)
        estimator_qnn = EstimatorQNN(
            circuit=isa_qc,
            observables=[isa_observables],
        ...


- 🔪 Passing gradients without a pass manager:

Some gradient algorithms may require creation of new circuits, and primitives from  
`qiskit-ibm-runtime` require transpilation. Please ensure a pass manager is also provided to gradients.

.. code:: ipython3
    
    # Wrong:
    ...
    pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend)
    gradient = ParamShiftEstimatorGradient(estimator=estimator)
    ...

    # Correct:
    ...
    pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend)
    gradient = ParamShiftEstimatorGradient(
        estimator=estimator, pass_manager=pass_manager
    )
    ...

- 🔪 Don't forget to migrate if you are using functions from `qiskit_algorithms` instead of `qiskit-machine-learning` for V2 primitives.
- 🔪 Some gradients such as SPSA and LCU from `qiskit_machine_learning.gradients` can be very prone to noise, be cautious of gradient values.
