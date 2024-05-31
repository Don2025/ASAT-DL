from torch import nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x[:, 0, :])))
    
net = MLPClassifier(input_size=768, hidden_size=128)
devices = try_all_gpus()
lr, num_epochs = 0.001, 30
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
train(net, train_loader, test_loader, loss, trainer, num_epochs, devices)