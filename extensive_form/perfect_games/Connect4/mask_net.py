import torch 
import torch.nn as nn
from torch.distributions import Categorical
import os
import numpy as np
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input, width, depth, out_num, path):
        super(MLP, self).__init__()
        self.critic = []
        self.actor = []
        self._path = path
        for i in range(depth+1):
            if i != depth:
                self.critic.append(nn.Linear(input, width))
                self.actor.append(nn.Linear(input, width))
                input = width
            else:
                self.critic.append(nn.Linear(input, 1))
                self.actor.append(nn.Linear(input, out_num))
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

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 7
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s.view(-1, 3,6,7)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(128, 3, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3*6*7, 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.conv1 = nn.Conv2d(128, 32, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(6*7*32, 2)
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 3*6*7)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 6*7*32)
        p = self.fc(p)
        log_p = self.logsoftmax(p)
        p = log_p.exp()
        return log_p, p, v


class Conv2d(nn.Module):
    def __init__(self, input, width, depth, out_num, path):
        super(Conv2d, self).__init__()
        self.depth = depth
        self._path = path
        self.conv = ConvBlock()
        for block in range(self.depth):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()

    # gpu level
    def forward(self, x):
        x = self.conv(x)
        for block in range(self.depth):
            x = getattr(self, "res_%i" % block)(x)
        log_p, p,v = self.outblock(x)
        dist = Categorical(p)
        return dist, v, log_p


    # cpu level
    def inference(self, x):
        x = torch.Tensor(x)
        x = self.conv(x)
        for block in range(self.depth):
            x = getattr(self, "res_%i" % block)(x)
        log_p, p,v = self.outblock(x)
        dist = Categorical(p)
        value = v.detach().numpy()

        return dist, value[0][0]

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
