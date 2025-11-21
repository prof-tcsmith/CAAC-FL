"""
Shared model architectures for all experiment levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-10 classification.

    Architecture:
    - Conv1: 3 -> 32 channels, 3x3 kernel
    - Conv2: 32 -> 64 channels, 3x3 kernel
    - FC1: 64*8*8 -> 128
    - FC2: 128 -> 10
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16

        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8

        # Flatten
        x = x.view(-1, 64 * 8 * 8)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_parameters(self):
        """Return model parameters as a list of tensors."""
        return [param.data.clone() for param in self.parameters()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of tensors."""
        for param, new_param in zip(self.parameters(), parameters):
            param.data = new_param.clone()

    def get_gradients(self):
        """Return model gradients as a list of tensors."""
        return [param.grad.clone() if param.grad is not None else torch.zeros_like(param)
                for param in self.parameters()]


class MLP(nn.Module):
    """
    Multi-layer Perceptron for simpler experiments.

    Architecture:
    - FC1: 3072 -> 512
    - FC2: 512 -> 256
    - FC3: 256 -> 10
    """

    def __init__(self, input_dim=3072, hidden_dims=[512, 256], num_classes=10):
        super(MLP, self).__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.model(x)

    def get_parameters(self):
        """Return model parameters as a list of tensors."""
        return [param.data.clone() for param in self.parameters()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of tensors."""
        for param, new_param in zip(self.parameters(), parameters):
            param.data = new_param.clone()

    def get_gradients(self):
        """Return model gradients as a list of tensors."""
        return [param.grad.clone() if param.grad is not None else torch.zeros_like(param)
                for param in self.parameters()]


def create_model(model_name='cnn', **kwargs):
    """
    Factory function to create models.

    Args:
        model_name: 'cnn' or 'mlp'
        **kwargs: Additional arguments passed to model constructor

    Returns:
        PyTorch model
    """
    if model_name.lower() == 'cnn':
        return SimpleCNN(**kwargs)
    elif model_name.lower() == 'mlp':
        return MLP(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("Testing SimpleCNN:")
    cnn = SimpleCNN()
    print(f"  Parameters: {count_parameters(cnn):,}")

    # Test forward pass
    dummy_input = torch.randn(4, 3, 32, 32)
    output = cnn(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

    print("\nTesting MLP:")
    mlp = MLP()
    print(f"  Parameters: {count_parameters(mlp):,}")
    output = mlp(dummy_input)
    print(f"  Output shape: {output.shape}")
