import math
import torch
import torch.nn as nn
import pennylane as qml


class QuantumVQCBranch(nn.Module):
    """
    8-qubit variational quantum circuit branch.

    We map the fused bilateral embedding into 8 rotation angles,
    run them through a PennyLane quantum circuit with entangling layers,
    and read out Pauli-Z expectation values as features for classification.

    The 2^8 = 256 dimensional Hilbert space gives us a richer feature
    separation than a purely classical head, especially for borderline
    co-morbid cases like DR+Hypertension.
    """

    def __init__(
        self,
        in_features: int = 512,
        n_qubits: int = 8,
        n_layers: int = 3,
        num_classes: int = 8,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # classical compression down to n_qubits angles
        self.pre_net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.SiLU(),
            nn.Linear(64, n_qubits),
            nn.Tanh(),
        )

        # learnable circuit weights: [layers, qubits, 3 euler angles]
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        self.post_net = nn.Linear(n_qubits, num_classes)

        # must run on CPU - MPS/CUDA not supported by default.qubit
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, weights):
            for j in range(n_qubits):
                qml.RY(inputs[j], wires=j)
            for l in range(n_layers):
                for j in range(n_qubits):
                    qml.Rot(weights[l, j, 0], weights[l, j, 1], weights[l, j, 2], wires=j)
                for j in range(n_qubits):
                    qml.CNOT(wires=[j, (j + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

        self.qnode = _circuit

    def forward(self, f_fused: torch.Tensor) -> torch.Tensor:
        angles = self.pre_net(f_fused) * math.pi

        # quantum sim needs CPU tensors
        a_cpu = angles.cpu()
        w_cpu = self.weights.cpu()

        results = []
        for i in range(angles.size(0)):
            ev = self.qnode(a_cpu[i], w_cpu)
            results.append(torch.stack(ev))

        z = torch.stack(results, dim=0).to(dtype=f_fused.dtype, device=f_fused.device)
        return self.post_net(z)
