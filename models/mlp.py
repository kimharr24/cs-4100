import torch

class MLP(torch.nn.Module):
    def __init__(self, embedding_dim: int, max_word_count: int = 20):
        super(MLP, self).__init__()

        self.fc1 = torch.nn.Linear(embedding_dim * max_word_count, 4056)
        self.fc2 = torch.nn.Linear(4056, 2048)
        self.fc3 = torch.nn.Linear(2048, 512)
        self.fc4 = torch.nn.Linear(512, 1)

        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x