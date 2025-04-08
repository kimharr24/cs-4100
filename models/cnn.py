import torch

class CNN(torch.nn.Module):
    def __init__(self, embedding_dim: int):
        super(CNN, self).__init__() 
        self.embedding_dim = embedding_dim

        self.conv1 = torch.nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3)
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