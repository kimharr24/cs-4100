import torch

class CNN(torch.nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super(CNN, self).__init__() 
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.conv1 = torch.nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3)
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)

        self.pool = torch.nn.MaxPool1d(kernel_size=2)
        self.relu = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(128 * ((embedding_dim - 2) // 2), 256)
        self.fc2 = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
            


