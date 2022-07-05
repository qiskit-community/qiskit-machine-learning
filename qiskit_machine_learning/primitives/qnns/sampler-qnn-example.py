# THIS EXAMPLE USES THE TERRA PRIMITIVES!
import numpy as np
from qiskit.primitives import Sampler, Estimator
from qiskit import Aer, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.neural_networks import CircuitQNN
from primitives.gradient.param_shift_sampler_gradient import ParamShiftSamplerGradient
from primitives.gradient.finite_diff_sampler_gradient import FiniteDiffSamplerGradient
algorithm_globals.random_seed = 42

# DEFINE CIRCUIT FOR SAMPLER
num_qubits = 3
qc = RealAmplitudes(num_qubits, entanglement="linear", reps=1)
qc.draw(output="mpl")
# ADD MEASUREMENT HERE --> TRICKY
qc.measure_all()

# ---------------------

from qiskit import Aer
from qiskit.utils import QuantumInstance

qi_qasm = QuantumInstance(Aer.get_backend("aer_simulator"), shots=10)
qi_sv = QuantumInstance(Aer.get_backend("statevector_simulator"))

parity = lambda x: "{:b}".format(x).count("1") % 2
output_shape = 2  # this is required in case of a callable with dense output

qnn2 = CircuitQNN(
    qc,
    input_params=qc.parameters[:3],
    weight_params=qc.parameters[3:],
    sparse = False,
    interpret = parity,
    output_shape = output_shape,
    quantum_instance=qi_sv,
)
inputs = np.asarray(algorithm_globals.random.random(size = (1, qnn2._num_inputs)))
weights = algorithm_globals.random.random(qnn2._num_weights)
print("inputs: ", inputs)
print("weights: ", weights)

np.set_printoptions(precision=2)
f = qnn2.forward(inputs, weights)
print( f"fwd pass: {f}")
np.set_printoptions(precision=2)
b = qnn2.backward(inputs, weights)
print( f"bkwd pass: {b}" )
# ---------------------

# IMPORT QNN
from neural_networks.sampler_qnn_2 import SamplerQNN

with SamplerQNN(
        circuit=qc,
        input_params=qc.parameters[:3],
        weight_params=qc.parameters[3:],
        sampler_factory=Sampler,
        interpret = parity,
        output_shape = output_shape,
    ) as qnn:
    # inputs = np.asarray(algorithm_globals.random.random((2, qnn._num_inputs)))
    # weights = algorithm_globals.random.random(qnn._num_weights)
    print("inputs: ", inputs)
    print("weights: ", weights)

    np.set_printoptions(precision=2)
    f = qnn.forward(inputs, weights)
    print(f"fwd pass: {f}")
    np.set_printoptions(precision=2)
    b = qnn.backward(inputs, weights)
    print(f"bkwd pass: {b}")


