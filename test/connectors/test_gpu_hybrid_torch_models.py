from qiskit_machine_learning import QiskitMachineLearningError
from test import gpu
from test.connectors.test_hybrid_torch_models import TestHybridTorchModels
from test.connectors.test_torch_connector import TestTorchConnector


@gpu
class TestGPUTorchConnector(TestHybridTorchModels):
    def _get_device(self):
        import torch

        if not torch.cuda.is_available():
            raise QiskitMachineLearningError("CUDA is not available")
        return torch.device("cuda")
