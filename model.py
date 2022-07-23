import torch 
import torch.nn as nn
from torch.distributions import Categorical

class MLP(nn.module):

  def __init__(self, width, depth, out_num, path):
    super(MLP, self).__init()

    self.critic = nn.Sequential()
    self.actor = nn.Sequential()

    self._path = path

    for i in range(depth+1):
      if i != depth:
        self.critic.add_module('mlp_%d' %i, nn.Linear(width, width))
        self.actor.add_module('mlp_%d' %i, nn.Linear(width, width))
      else:
        self.critic.add_module('mlp_%d' %i, nn.Linear(width, 1))
        self.actor.add_module('mlp_%d' %i, nn.Linear(width, out_num))
        self.actor.add_module('mlp_%d' %i, nn.Softmax(dim=1))
      if i != depth:
        self.critic.add_module('relu_%d' %i, nn.ReLu())
        self.actor.add_module('relu_%d' %i, nn.ReLu())

    # gpu level
    def forward(self, x):
        values = self.critic(x)
        probs = self.actor(x)
        dis = Categorical(probs)

        return dist, value

    # cpu level
    def inference(self, x):
        x = np.expand_dim(x, 0)
        x = torch.Tensor(x)
        value = self.critic(x)
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