"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import numpy as np
from torch.distributions import Categorical
import torch
import torch.nn.functional as F
from torch import nn

class LandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(319 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        # policy-value 
        self.policy = nn.Linear(512, 2)
        self.value =  nn.Linear(512, 1)

    def forward(self, z, x):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)

        values = torch.tanh(self.value(x))
        probs = F.softmax(self.policy(x), dim=1)
        dist = Categorical(probs)
        return dist, values

    def inference(self, z, x):
        with torch.no_grad():
            z, x = torch.unsqueeze(z, dim=0), torch.unsqueeze(x, dim=0)
            lstm_out, (h_n, _) = self.lstm(z)
            lstm_out = lstm_out[:,-1,:]
            x = torch.cat([lstm_out,x], dim=-1)
            x = self.dense1(x)
            x = torch.relu(x)
            x = self.dense2(x)
            x = torch.relu(x)
            x = self.dense3(x)
            x = torch.relu(x)
            x = self.dense4(x)
            x = torch.relu(x)
            x = self.dense5(x)
            x = torch.relu(x)

            values = torch.tanh(self.value(x))
            probs = F.softmax(self.policy(x), dim=1)
            dist = Categorical(probs)
            return dist, values


class FarmerLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(430 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        # policy-value 
        self.policy = nn.Linear(512, 2)
        self.value =  nn.Linear(512, 1)

    def forward(self, z, x):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        print(x.shape)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)

        values = torch.tanh(self.value(x))
        probs = F.softmax(self.policy(x), dim=1)
        dist = Categorical(probs)
        return dist, values

    def inference(self, z, x):
        with torch.no_grad():
            z, x = torch.unsqueeze(z, dim=0), torch.unsqueeze(x, dim=0)
            lstm_out, (h_n, _) = self.lstm(z)
            lstm_out = lstm_out[:,-1,:]
            x = torch.cat([lstm_out,x], dim=-1)
            x = self.dense1(x)
            x = torch.relu(x)
            x = self.dense2(x)
            x = torch.relu(x)
            x = self.dense3(x)
            x = torch.relu(x)
            x = self.dense4(x)
            x = torch.relu(x)
            x = self.dense5(x)
            x = torch.relu(x)

            values = torch.tanh(self.value(x))
            probs = F.softmax(self.policy(x), dim=1)
            dist = Categorical(probs)
            return dist, values

class MaskNet:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0, position='landlord_down'):
        self.model = None
        self.position = position
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.device = torch.device(device)
        if position == 'landlord':
            self.model = LandlordLstmModel().to(torch.device(device))
        else:
            self.model = FarmerLstmModel().to(torch.device(device))

    def forward(self, z, x):
        return self.model.forward(z, x)

    def inference(self, z, x):
        return self.model.inference(z, x)

    def share_memory(self):
        self.model.share_memory()

    def eval(self):
        self.model.eval()

    def get_model(self):
        return self.model

    def parameters(self):
        return self.model.parameters()
