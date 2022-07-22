import torch 
import torch.nn as nn
from torch.distributions import Categorical

class MLP(nn.module):

  def __init__(self, num_inputs, outputs, hidden_size, path):
    super(MLP, self).__init()

    self.critic = nn.Sequential()
    self.actor = nn.Sequential()

    self._path = path

    for i in range(len(hidden_size)):
      if i == 0:
        self.critic.add_module('mlp_%d' %i, nn.Linear(num_inputs, hidden_size[i]))
        self.actor.add_module('mlp_%d' %i, nn.Linear(num_inputs, hidden_size[i]))
      elif i != len - 1:
        self.critic.add_module('mlp_%d' %i, nn.Linear(hidden_size[i-1], hidden_size[i]))
        self.actor.add_module('mlp_%d' %i, nn.Linear(hidden_size[i-1], hidden_size[i]))
      else:
        self.critic.add_module('mlp_%d' %i, nn.Linear(hidden_size[i], 1))
        self.actor.add_module('mlp_%d' %i, nn.Linear(hidden_size[i], outputs))
        self.actor.add_module('mlp_%d' %i, nn.Softmax(dim=1))

      if i != len - 1:
        self.critic.add_module('relu_%d' %i, nn.ReLu())
        self.actor.add_module('relu_%d' %i, nn.ReLu())

    # gpu level
    def forward(self, x):
        values = self.critic(x)
        probs = self.actor(x)
        dis = Categorical(probs)

        return dist, value

    # cpu level
    def inference(self, obs):
        obs = np.expand_dim(obs, 0)
        obs = torch.Tensor(obs)
        value = self.critic(obs)
        probs = self.actor(x)
        probs = torch.squeeze(x, 0)
        dis = Categorical(probs)

        value = value.detach().numpy()

        return dis, value[0][0]

    def save_checkpoint(self, step):
        name = 'model_%d.pth' %(step)
        save_path = os.path.join(self._path, "checkpoints", name)
        torch.save(self.state_dict(), name)
        return save_path

    # cpu level
    def load_checkppoint(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc:s storage)
        self.load_state_dict(checkpoint['state_dict'])

