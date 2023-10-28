import torch 
import torch.nn as nn
from torch.distributions import Categorical
import os
import numpy as np
from math import isnan

class MLP(nn.Module):
    def __init__(self, input, width, depth, out_num, path):
        super(MLP, self).__init__()
        self.critic = []
        self.actor = []
        self._path = path
        self.depth = depth
        for i in range(depth+1):
            if i != depth:
                self.critic.append(nn.Linear(input, width))
                self.actor.append(nn.Linear(input, width))
                input = width
            else:
                self.critic.append(nn.Linear(input, 1))
                self.actor.append(nn.Linear(input, out_num))
                #self.actor.append(nn.Softmax(dim=1))
            if i != depth:
                self.critic.append(nn.ReLU())
                self.actor.append(nn.ReLU())
        self.critic = nn.Sequential(*self.critic)
        self.actor = nn.Sequential(*self.actor)
    # gpu level
    def forward(self, x):
        values = self.critic(x)
        output = self.actor(x)
        softmax = nn.Softmax(dim=1)
        log_softmax = nn.LogSoftmax(dim=1)
        probs = softmax(output)
        log_probs = log_softmax(output)

        try:
          dist = Categorical(probs)
        except:
          print(x)
          print(probs)

        return dist, values, log_probs

    # cpu level
    def inference(self, x):
        x = np.expand_dims(x, 0)
        x = torch.Tensor(x)
        value = self.critic(x)
        output = self.actor(x)
        softmax = nn.Softmax(dim=1)
        probs = softmax(output)
        dis = Categorical(probs)
        value = value.detach().numpy()

        return dis, value[0][0]

    def save_checkpoint(self, step):
        name = 'model_%d.pth' %(step)
        save_path = os.path.join(self._path, "checkpoints", name)
        torch.save(self.state_dict(), save_path)
        return save_path

    # cpu level
    def load_checkpoint(self, path):
        #print("path location", path)
        checkpoint = torch.load(path, map_location=lambda storage, loc:storage)
        self.load_state_dict(checkpoint)
