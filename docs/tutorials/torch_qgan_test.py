# Necessary imports

import numpy as np

from torch import Tensor, stack, reshape, FloatTensor
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F


from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.opflow import Gradient, StateFn
from qiskit.circuit.library import TwoLocal
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.datasets.dataset_helper import discretize_and_truncate

# Set seed for random generators
algorithm_globals.random_seed = 42

# TODO: update
data_dim = [2, 2]

training_data = np.random.default_rng().multivariate_normal(mean=[0., 0.], cov=[[1, 0], [0, 1]], size=1000, check_valid='warn',
                                                        tol=1e-8, method='svd')


batch_size = 100

dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

# declare quantum instance
backend = Aer.get_backend('aer_simulator')
qi = QuantumInstance(backend, shots = batch_size)


##### Data
bounds_min = np.percentile(training_data, 5, axis=0)
bounds_max = np.percentile(training_data, 95, axis=0)
bounds = []
for i, _ in enumerate(bounds_min):
    bounds.append([bounds_min[i], bounds_max[i]])

(data,
data_grid,
grid_elements,
prob_data ) = discretize_and_truncate(
training_data,
np.array(bounds),
data_dim,
return_data_grid_elements=True,
return_prob=True,
prob_non_zero=True,
)

########### Generator

qnn = QuantumCircuit(sum(data_dim))
qnn.h(qnn.qubits)
ansatz = TwoLocal(sum(data_dim), "ry", "cz", reps=2, entanglement="circular")
qnn.compose(ansatz, inplace=True)

# Pick gradient method
grad_method = 'param_shift'

def generator_():
        circuit_qnn = CircuitQNN(qnn, input_params=[], weight_params = ansatz.ordered_parameters,
                                 quantum_instance=qi, sampling=True, sparse=False,
                                 interpret=lambda x: grid_elements[x], input_gradients=True,) # gradient=Gradient(),

        return TorchConnector(circuit_qnn)

def generator_grad(param_values):
    grad = Gradient(grad_method=grad_method).gradient_wrapper(StateFn(qnn), ansatz.ordered_parameters, backend=qi)
    grad_values = grad(param_values)
    return grad_values.tolist()

####### Discriminator

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.Linear_in = nn.Linear(len(data_dim), 51)
        self.Leaky_ReLU = nn.LeakyReLU(0.2, inplace=True)
        self.Linear51 = nn.Linear(51, 26)
        self.Linear26 = nn.Linear(26, 1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.Linear_in(input)
        x = self.Leaky_ReLU(x)
        x = self.Linear51(x)
        x = self.Leaky_ReLU(x)
        x = self.Linear26(x)
        x = self.Sigmoid(x)
        return x


###### Optimizer

# Loss function
g_loss_fun = nn.BCELoss()
d_loss_fun = nn.BCELoss()


#TODO overwrite PyTorch BCELoss gradient?
def g_loss_fun_grad(param_values, discriminator_):
    """

    """
    grads = generator_grad(param_values)
    loss_grad = ()
    for j, grad in enumerate(grads):
        cx = grad[0].tocoo()
        input = []
        target = []
        weight = []
        for index, prob_grad in zip(cx.col, cx.data):
            input.append(grid_elements[index])
            target.append([1.])
            weight.append([prob_grad])
        bce_loss_grad = F.binary_cross_entropy(discriminator_(Tensor(input)), Tensor(target), weight=Tensor(weight))
        loss_grad += (bce_loss_grad, )
    loss_grad = stack(loss_grad)
    return loss_grad

# Initialize generator and discriminator
generator = generator_()
discriminator = Discriminator()

lr=0.0002
b1=0.5
b2=0.999
n_epochs=100

#TODO generator.parameters() replace with PyTorch parameter object

optimizer_G = Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))


######## Training

for epoch in range(n_epochs):
    for i, data in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(data.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(data.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_data = Variable(data.type(Tensor))
        # Generate a batch of images
        gen_data = generator()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        disc_data = discriminator(real_data)
        real_loss = d_loss_fun(disc_data, valid)

        fake_loss = d_loss_fun(discriminator(gen_data), fake)  # (discriminator(gen_data).detach(), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward(retain_graph=True)
        print('discriminator grad ', discriminator.Linear26.weight.grad)
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # print('Test gradients ', generator_grad(generator.weight.data.numpy()))
        # print('Test gradients loss ', g_loss_fun_grad(generator.weight.data.numpy(), discriminator))
        # gen_data.grad =  generator_grad(generator.weight.data.numpy())

        # # Loss measures generator's ability to fool the discriminator
        g_loss = g_loss_fun(discriminator(gen_data), valid)
        g_loss.retain_grad = True
        g_loss_grad = g_loss_fun_grad(generator.weight.data.numpy(), discriminator)
        g_loss.backward(retain_graph=True) # TODO gradient=Tensor([1.])
        for j, param in enumerate(generator.parameters()):
            param.grad = g_loss_grad
        optimizer_G.step()


        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
