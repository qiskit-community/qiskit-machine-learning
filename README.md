# Qiskit Machine Learning

[![License](https://img.shields.io/github/license/Qiskit/qiskit-machine-learning.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)<!--- long-description-skip-begin -->[![Build Status](https://github.com/Qiskit/qiskit-machine-learning/workflows/Machine%20Learning%20Unit%20Tests/badge.svg?branch=main)](https://github.com/Qiskit/qiskit-machine-learning/actions?query=workflow%3A"Machine%20Learning%20Unit%20Tests"+branch%3Amain+event%3Apush)[![](https://img.shields.io/github/release/Qiskit/qiskit-machine-learning.svg?style=popout-square)](https://github.com/Qiskit/qiskit-machine-learning/releases)[![](https://img.shields.io/pypi/dm/qiskit-machine-learning.svg?style=popout-square)](https://pypi.org/project/qiskit-machine-learning/)[![Coverage Status](https://coveralls.io/repos/github/Qiskit/qiskit-machine-learning/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-machine-learning?branch=main)<!--- long-description-skip-end -->

Qiskit Machine Learning introduces fundamental computational building blocks - such as Quantum Kernels
and Quantum Neural Networks - used in different applications, including classification and regression.
On the one hand, this design is very easy to use and allows users to rapidly prototype a first model
without deep quantum computing knowledge. On the other hand, Qiskit Machine Learning is very flexible,
and users can easily extend it to support cutting-edge quantum machine learning research.

Qiskit Machine Learning provides the
[FidelityQuantumKernel](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.kernels.QuantumKernel.html#qiskit_machine_learning.kernels.FidelityQuantumKernel)
class that makes use of the [Fidelity](https://qiskit.org/documentation/stubs/qiskit.algorithms.state_fidelities.BaseStateFidelity.html) algorithm introduced in Qiskit and can be easily used to directly compute
kernel matrices for given datasets or can be passed to a Quantum Support Vector Classifier 
[QSVC](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.algorithms.QSVC.html#qiskit_machine_learning.algorithms.QSVC) or
Quantum Support Vector Regressor
[QSVR](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.algorithms.QSVR.html#qiskit_machine_learning.algorithms.QSVR)
to quickly start solving classification or regression problems.
It also can be used with many other existing kernel-based machine learning algorithms from established
classical frameworks.

Qiskit Machine Learning defines a generic interface for neural networks that is implemented by different
quantum neural networks. Two core implementations are readily provided, such as the
[EstimatorQNN](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.neural_networks.EstimatorQNN.html),
and the [SamplerQNN](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.neural_networks.SamplerQNN.html).
The [EstimatorQNN](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.neural_networks.EstimatorQNN.html)
leverages the [Estimator](https://qiskit.org/documentation/stubs/qiskit.primitives.BaseEstimator.html) primitive from Qiskit and 
allows users to combine parametrized quantum circuits with quantum mechanical observables. The circuits can be constructed using, for example, building blocks
from Qiskit’s circuit library, and the QNN’s output is given by the expected value of the observable.
The [SamplerQNN](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.neural_networks.SamplerQNN.html)
leverages another primitive introduced in Qiskit, the [Sampler](https://qiskit.org/documentation/stubs/qiskit.primitives.BaseSampler.html) primitive. 
This neural network translates quasi-probabilities of bitstrings estimated by the primitive into a desired output. This 
translation step can be used to interpret a given bitstring in a particular context, e.g. translating it into a set of classes.

The neural networks include the functionality to evaluate them for a given input as well as to compute the
corresponding gradients, which is important for efficient training. To train and use neural networks,
Qiskit Machine Learning provides a variety of learning algorithms such as the
[NeuralNetworkClassifier](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.algorithms.NeuralNetworkClassifier.html#qiskit_machine_learning.algorithms.NeuralNetworkClassifier)
and
[NeuralNetworkRegressor](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.algorithms.NeuralNetworkRegressor.html#qiskit_machine_learning.algorithms.NeuralNetworkRegressor).
Both take a QNN as input and then use it in a classification or regression context.
To allow an easy start, two convenience implementations are provided - the Variational Quantum Classifier
[VQC](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.algorithms.VQC.html#qiskit_machine_learning.algorithms.VQC)
as well as the Variational Quantum Regressor
[VQR](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.algorithms.VQR.html#qiskit_machine_learning.algorithms.VQR).
Both take just a feature map and an ansatz and construct the underlying QNN automatically.

In addition to the models provided directly in Qiskit Machine Learning, it has the
[TorchConnector](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.connectors.TorchConnector.html#qiskit_machine_learning.connectors.TorchConnector),
which allows users to integrate all of our quantum neural networks directly into the
[PyTorch](https://pytorch.org)
open source machine learning library. Thanks to Qiskit’s gradient algorithms, this includes automatic
differentiation - the overall gradients computed by [PyTorch](https://pytorch.org)
during the backpropagation take into
account quantum neural networks, too. The flexible design also allows the building of connectors
to other packages in the future.

## Installation

We encourage installing Qiskit Machine Learning via the pip tool (a python package manager).

```bash
pip install qiskit-machine-learning
```

**pip** will handle all dependencies automatically and you will always install the latest
(and well-tested) version.

If you want to work on the very latest work-in-progress versions, either to try features ahead of
their official release or if you want to contribute to Machine Learning, then you can install from source.
To do this follow the instructions in the
 [documentation](https://qiskit.org/documentation/machine-learning/getting_started.html#installation).


----------------------------------------------------------------------------------------------------

### Optional Installs

* **PyTorch**, may be installed either using command `pip install 'qiskit-machine-learning[torch]'` to install the
  package or refer to PyTorch [getting started](https://pytorch.org/get-started/locally/). When PyTorch
  is installed, the `TorchConnector` facilitates its use of quantum computed networks.

* **Sparse**, may be installed using command `pip install 'qiskit-machine-learning[sparse]'` to install the
  package. Sparse being installed will enable the usage of sparse arrays/tensors.

### Creating Your First Machine Learning Programming Experiment in Qiskit

Now that Qiskit Machine Learning is installed, it's time to begin working with the Machine Learning module.
Let's try an experiment using VQC (Variational Quantum Classifier) algorithm to
train and test samples from a data set to see how accurately the test set can
be classified.

```python
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit.utils import algorithm_globals

from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.datasets import ad_hoc_data

seed = 1376
algorithm_globals.random_seed = seed

# Use ad hoc data set for training and test data
feature_dim = 2  # dimension of each data point
training_size = 20
test_size = 10

# training features, training labels, test features, test labels as np.ndarray,
# one hot encoding for labels
training_features, training_labels, test_features, test_labels = ad_hoc_data(
    training_size=training_size, test_size=test_size, n=feature_dim, gap=0.3
)

feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement="linear")
ansatz = TwoLocal(feature_map.num_qubits, ["ry", "rz"], "cz", reps=3)
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=100),
)
vqc.fit(training_features, training_labels)

score = vqc.score(test_features, test_labels)
print(f"Testing accuracy: {score:0.2f}")
```

### Further examples

Learning path notebooks may be found in the
[Machine Learning tutorials](https://qiskit.org/documentation/machine-learning/tutorials/index.html) section
of the documentation and are a great place to start. 

Another good place to learn the fundamentals of quantum machine learning is the
[Quantum Machine Learning](https://learn.qiskit.org/course/machine-learning/introduction) course 
on the Qiskit Textbook's website. The course is very convenient for beginners who are eager to learn 
quantum machine learning from scratch, as well as understand the background and theory behind algorithms in
Qiskit Machine Learning. The course covers a variety of topics to build understanding of parameterized
circuits, data encoding, variational algorithms etc., and in the end the ultimate goal of machine
learning - how to build and train quantum ML models for supervised and unsupervised learning. 
The textbook course is complementary to the tutorials of this module, where the tutorials focus
on actual Qiskit Machine Learning algorithms, the course more explains and details underlying fundamentals
of quantum machine learning.

----------------------------------------------------------------------------------------------------

## Contribution Guidelines

If you'd like to contribute to Qiskit, please take a look at our
[contribution guidelines](https://github.com/Qiskit/qiskit-machine-learning/blob/main/CONTRIBUTING.md).
This project adheres to Qiskit's [code of conduct](https://github.com/Qiskit/qiskit-machine-learning/blob/main/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-machine-learning/issues) for tracking requests and bugs. Please
[join the Qiskit Slack community](https://qisk.it/join-slack)
and for discussion and simple questions.
For questions that are more suited for a forum, we use the **Qiskit** tag in [Stack Overflow](https://stackoverflow.com/questions/tagged/qiskit).

## Authors and Citation

Machine Learning was inspired, authored and brought about by the collective work of a team of researchers.
Machine Learning continues to grow with the help and work of
[many people](https://github.com/Qiskit/qiskit-machine-learning/graphs/contributors), who contribute
to the project at different levels.
If you use Qiskit, please cite as per the provided
[BibTeX file](https://github.com/Qiskit/qiskit/blob/master/Qiskit.bib).

Please note that if you do not like the way your name is cited in the BibTex file then consult
the information found in the [.mailmap](https://github.com/Qiskit/qiskit-machine-learning/blob/main/.mailmap)
file.

## License

This project uses the [Apache License 2.0](https://github.com/Qiskit/qiskit-machine-learning/blob/main/LICENSE.txt).
