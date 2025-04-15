import torch


class CNNFromScratch(torch.nn.Module):
    """CNN model implemented from scratch for performing 1D convolutions on text data."""
    def __init__(self, embedding_dim: int):
        super(CNNFromScratch, self).__init__()
        self.embedding_dim = embedding_dim

        self.kernel_1 = torch.randn((64, embedding_dim, 3), requires_grad=True)
        self.kernel_2 = torch.randn((128, 64, 5), requires_grad=True)
        self.kernel_3 = torch.randn((256, 128, 7), requires_grad=True)

        self.mlp_weight_1 = torch.randn((1024, 256), requires_grad=True)
        self.mlp_weight_2 = torch.randn((256, 128), requires_grad=True)
        self.mlp_weight_3 = torch.randn((128, 1), requires_grad=True)

        self.relu = torch.nn.ReLU()

    def pool1d(
        self, x: torch.Tensor, kernel_size: int = 2, stride: int = 2
    ) -> torch.Tensor:
        """Performs a 1D max pooling operation."""
        batch_size, embedding_dim, width = x.shape
        output_width = (width - kernel_size) // stride + 1

        pooled_output = torch.zeros((batch_size, embedding_dim, output_width))
        for i in range(output_width):
            start_idx = i * stride
            end_idx = start_idx + kernel_size

            x_slice = x[:, :, start_idx:end_idx]
            pooled_output[:, :, i] = torch.max(x_slice, dim=2).values

        return pooled_output

    def conv1d(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Performs a 1D convolution operation."""
        batch_size, _, width = x.shape
        out_channels, _, kernel_size = kernel.shape
        output_width = width - kernel_size + 1

        output = torch.zeros((batch_size, out_channels, output_width))
        for i in range(output_width):
            x_slice = x[:, :, i : i + kernel_size]
            for j in range(out_channels):
                output[:, j, i] = torch.sum(x_slice * kernel[j, :, :], dim=(1, 2))

        return output

    def forward(self, x):
        x = self.relu(self.conv1d(x, self.kernel_1))
        x = self.relu(self.conv1d(x, self.kernel_2))
        x = self.relu(self.conv1d(x, self.kernel_3))

        x = self.pool1d(x)
        x = x.view(x.size(0), -1)

        x = self.relu(torch.matmul(x, self.mlp_weight_1))
        x = self.relu(torch.matmul(x, self.mlp_weight_2))
        x = torch.matmul(x, self.mlp_weight_3)
        return x


class CNN(torch.nn.Module):
    """Standard CNN built with PyTorch modules."""
    def __init__(self, embedding_dim: int):
        super(CNN, self).__init__()
        self.embedding_dim = embedding_dim

        self.conv1 = torch.nn.Conv1d(
            in_channels=embedding_dim, out_channels=64, kernel_size=3
        )
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5)
        self.conv3 = torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7)

        self.pool = torch.nn.MaxPool1d(kernel_size=2)
        self.relu = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(1024, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    model = CNNFromScratch(embedding_dim=300)
    sample_input = torch.rand((32, 300, 20))

    print(model(sample_input).shape)
