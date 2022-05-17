import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import VQC, NeuralNetworkClassifier
from qiskit import Aer




# Choose a seed for the quantum instance initialisation.
seed = 123
num_qubits = 2
num_features = 2
loss="cross_entropy"

# Construct toy data set.
y = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# y = np.asarray(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
# y = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
# y = np.asarray(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C"])

y = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# y = np.array([y, 1 - y]).transpose()

onehot_target = np.zeros((y.size, int(y.max() + 1)))
onehot_target[np.arange(y.size), y.astype(int)] = 1

x = np.random.rand(len(y), num_features)

# svc = SVC(kernel=x"linear", C=0.025)
# svc.fit(x, target_one_hot)

# Initialise the VQC.
backend = Aer.get_backend("aer_simulator_statevector")
qin = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)
vqc = VQC(
    num_qubits=num_qubits,
    loss=loss,
    quantum_instance=qin,
)

# # Fit the VQC to the constructed toy x
vqc.fit(x, y)

score = vqc.score(x, y)
print(score)







