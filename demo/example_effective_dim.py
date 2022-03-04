from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
import matplotlib.pyplot as plt
import numpy as np
import itertools

from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit.utils import QuantumInstance
from qiskit import Aer, QuantumCircuit

from qiskit_machine_learning.algorithms.effective_dimension import EffectiveDimension, LocalEffectiveDimension
# This is an example file to create a quantum model and compute its effective dimension

# create ranges for the number of data, n
n = [5000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000]

# number of times to repeat the variational circuit
blocks = 1

# number of qubits, data samples and parameter sets to estimate the effective dimension
num_qubits = 3
num_inputs = 10
num_thetas = 10

# create a feature map
feat_map = ZFeatureMap(feature_dimension=num_qubits, reps=1)

# create a variational circuit
ansatz = RealAmplitudes(num_qubits, reps=blocks)

# define quantum instance (statevector)
qi_sv = QuantumInstance(Aer.get_backend("aer_simulator_statevector"))

qc = QuantumCircuit(feat_map.num_qubits)
qc.append(feat_map, range(feat_map.num_qubits))
qc.append(ansatz, range(ansatz.num_qubits))

# parity maps bitstrings to 0 or 1
def parity(x):
    return "{:b}".format(x).count("1") % 2
output_shape = 2  # corresponds to the number of classes,

# construct QNN
qnn = CircuitQNN(
    qc,
    input_params=feat_map.parameters,
    weight_params=ansatz.parameters,
    interpret=parity,
    output_shape=output_shape,
    sparse=False,
    quantum_instance=qi_sv
)

from qiskit_machine_learning.neural_networks import OpflowQNN, TwoLayerQNN
from qiskit.opflow import StateFn, PauliSumOp, AerPauliExpectation, ListOp, Gradient

# specify the observable
observable = PauliSumOp.from_list([("Z" * num_qubits, 1)])
print(observable)
# define two layer QNN
qnn3 = TwoLayerQNN(
    num_qubits, feature_map=feat_map, ansatz=ansatz, observable=observable, quantum_instance=qi_sv
)

ed = EffectiveDimension(qnn3, num_thetas=num_thetas, num_inputs=num_inputs)
# f, trace = ed.get_fhat()
# print("f: ", f)
effdim = ed.eff_dim(n)

d = ed.d

print("effdim: ", effdim)
# plot the normalised effective dimension for the model
plt.plot(n, np.array(effdim[0])/d)
plt.xlabel('number of data')
plt.ylabel('normalised effective dimension')
plt.show()

