"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import numpy as np
from torch.distributions import Categorical
import torch
import torch.nn.functional as F
from torch import nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class LandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)

        self.policy_network = []
        self.value_network = []
        # policy network
        self.policy_network.append(layer_init(nn.Linear(319 + 128, 512)))
        self.policy_network.append(nn.Tanh())
        self.policy_network.append(layer_init(nn.Linear(512, 512)))
        self.policy_network.append(nn.Tanh())
        self.policy_network.append(layer_init(nn.Linear(512, 512)))
        self.policy_network.append(nn.Tanh())
        self.policy_network.append(layer_init(nn.Linear(512, 512)))
        self.policy_network.append(nn.Tanh())
        self.policy_network.append(layer_init(nn.Linear(512, 256)))
        self.policy_network.append(nn.Tanh())
        self.policy_network.append(layer_init(nn.Linear(256, 2), std=1.0))

        # value network
        self.value_network.append(layer_init(nn.Linear(319 + 128, 512)))
        self.value_network.append(nn.Tanh())
        self.value_network.append(layer_init(nn.Linear(512, 512)))
        self.value_network.append(nn.Tanh())
        self.value_network.append(layer_init(nn.Linear(512, 512)))
        self.value_network.append(nn.Tanh())
        self.value_network.append(layer_init(nn.Linear(512, 512)))
        self.value_network.append(nn.Tanh())
        self.value_network.append(layer_init(nn.Linear(512, 256)))
        self.value_network.append(nn.Tanh())
        self.value_network.append(layer_init(nn.Linear(256, 1), std=1.0))

        self.value_network = nn.Sequential(*self.value_network)
        self.policy_network = nn.Sequential(*self.policy_network)


    def forward(self, z, x):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        
        values = self.value_network(x)
        logits = self.policy_network(x)

        probs = F.softmax(logits, dim=1)
        log_prb = F.log_softmax(logits, dim=1)

        dist = Categorical(probs)
        return dist, values, log_prob

    def inference(self, z, x):
        with torch.no_grad():
            z, x = torch.unsqueeze(z, dim=0), torch.unsqueeze(x, dim=0)
            lstm_out, (h_n, _) = self.lstm(z)
            lstm_out = lstm_out[:,-1,:]
            x = torch.cat([lstm_out,x], dim=-1)
        
            values = self.value_network(x)
            probs = F.softmax(self.policy_network(x), dim=1)
            
            dist = Categorical(probs)
            return dist, values


class FarmerLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)

        self.policy_network = []
        self.value_network = []
        # policy network
        self.policy_network.append(layer_init(nn.Linear(430 + 128, 512)))
        self.policy_network.append(nn.Tanh)
        self.policy_network.append(layer_init(nn.Linear(512, 512)))
        self.policy_network.append(nn.Tanh)
        self.policy_network.append(layer_init(nn.Linear(512, 512)))
        self.policy_network.append(nn.Tanh)
        self.policy_network.append(layer_init(nn.Linear(512, 512)))
        self.policy_network.append(nn.Tanh)
        self.policy_network.append(layer_init(nn.Linear(512, 256)))
        self.policy_network.append(nn.Tanh)
        self.policy_network.append(layer_init(nn.Linear(256, 2), std=1.0))

        # value network
        self.value_network.append(layer_init(nn.Linear(430 + 128, 512)))
        self.value_network.append(nn.Tanh)
        self.value_network.append(layer_init(nn.Linear(512, 512)))
        self.value_network.append(nn.Tanh)
        self.value_network.append(layer_init(nn.Linear(512, 512)))
        self.value_network.append(nn.Tanh)
        self.value_network.append(layer_init(nn.Linear(512, 512)))
        self.value_network.append(nn.Tanh)
        self.value_network.append(layer_init(nn.Linear(512, 256)))
        self.value_network.append(nn.Tanh)
        self.value_network.append(layer_init(nn.Linear(256, 1), std=1.0))

        self.value_network = nn.Sequential(*self.value_network)
        self.policy_network = nn.Sequential(*self.policy_network)

    def forward(self, z, x):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)



        values = self.value_network(x)
        logits = self.policy_network(x)

        probs = F.softmax(logits, dim=1)
        log_prb = F.log_softmax(logits, dim=1)

        dist = Categorical(probs)
        return dist, values, log_prob
        
    def inference(self, z, x):
        with torch.no_grad():
            z, x = torch.unsqueeze(z, dim=0), torch.unsqueeze(x, dim=0)
            lstm_out, (h_n, _) = self.lstm(z)
            lstm_out = lstm_out[:,-1,:]
            x = torch.cat([lstm_out,x], dim=-1)

            values = self.value_network(x)
            probs = F.softmax(self.policy_network(x), dim=1)

            dist = Categorical(probs)
           
            return dist, values

class MaskNet:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0, position='landlord'):
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
