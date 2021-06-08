# Qiskit Machine Learning

[![License](https://img.shields.io/github/license/Qiskit/qiskit-machine-learning.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)[![Build Status](https://github.com/Qiskit/qiskit-machine-learning/workflows/Machine%20Learning%20Unit%20Tests/badge.svg?branch=main)](https://github.com/Qiskit/qiskit-machine-learning/actions?query=workflow%3A"Machine%20Learning%20Unit%20Tests"+branch%3Amain+event%3Apush)[![](https://img.shields.io/github/release/Qiskit/qiskit-machine-learning.svg?style=popout-square)](https://github.com/Qiskit/qiskit-machine-learning/releases)[![](https://img.shields.io/pypi/dm/qiskit-machine-learning.svg?style=popout-square)](https://pypi.org/project/qiskit-machine-learning/)[![Coverage Status](https://coveralls.io/repos/github/Qiskit/qiskit-machine-learning/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-machine-learning?branch=main)

The Machine Learning package simply contains sample datasets at present. It has some
classification algorithms such as QSVM and VQC (Variational Quantum Classifier), where this data
can be used for experiments, and there is also QGAN (Quantum Generative Adversarial Network)
algorithm.

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
 [documentation](https://qiskit.org/documentation/contributing_to_qiskit.html#installing-from-source).


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
        from qiskit import BasicAer
        from qiskit.utils import QuantumInstance, algorithm_globals
        from qiskit.algorithms.optimizers import COBYLA
        from qiskit.circuit.library import TwoLocal
        from qiskit_machine_learning.algorithms import VQC
        from qiskit_machine_learning.datasets import wine
        from qiskit_machine_learning.circuit.library import RawFeatureVector

        seed = 1376
        algorithm_globals.random_seed = seed

        # Use Wine data set for training and test data
        feature_dim = 4  # dimension of each data point
        training_size = 12
        test_size = 4

        # training features, training labels, test features, test labels as np.array,
        # one hot encoding for labels
        training_features, training_labels, test_features, test_labels = \
            wine(training_size=training_size, test_size=test_size, n=feature_dim)

        feature_map = RawFeatureVector(feature_dimension=feature_dim)
        ansatz = TwoLocal(feature_map.num_qubits, ['ry', 'rz'], 'cz', reps=3)
        vqc = VQC(feature_map=feature_map,
                  ansatz=ansatz,
                  optimizer=COBYLA(maxiter=100),
                  quantum_instance=QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                                   shots=1024,
                                                   seed_simulator=seed,
                                                   seed_transpiler=seed)
                  )
        vqc.fit(training_features, training_labels)

        score = vqc.score(test_features, test_labels)
        print('Testing accuracy: {:0.2f}'.format(score))
```

### Further examples

Learning path notebooks may be found in the
[Machine Learning tutorials](https://qiskit.org/documentation/machine-learning/tutorials) section
of the documentation and are a great place to start.

----------------------------------------------------------------------------------------------------

## Contribution Guidelines

If you'd like to contribute to Qiskit, please take a look at our
[contribution guidelines](./CONTRIBUTING.md).
This project adheres to Qiskit's [code of conduct](./CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-machine-learning/issues) for tracking requests and bugs. Please
[join the Qiskit Slack community](https://ibm.co/joinqiskitslack)
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

This project uses the [Apache License 2.0](LICENSE.txt).


