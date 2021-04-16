import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch 

# nn.Module = Base class for all neural network modules.
class LinearClassifier(nn.Module): 
        def __init__(self, lr, n_classes, input_dims):
        super().__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss() #nn.MSELoss() for RL

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, data):
        layer1 = F.sigmoid(self.fc1(data))
        layer2 = F.sigmoid(self.fc2(layer1))
        layer3 = self.fc3(layer2)

        return layer3 

    def learn(self, data, labels):
        self.optimizer.zero_grad() # zero out gradient from previous backward()
        data = torch.tensor(data).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        
        predictions = self.forward(data)

        cost = self.loss(predictions, labels)

        cost.backward() # accumulates gradient by addition for each parameter
        self.optimizer.step() # does the update step with the calculated gradients
        