import gymnasium as gym
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from DQN import DQN

print("Libraries Imported") 
#======================================================
# Control Panel
#======================================================
m = nn.Softmax(dim=0)
a = torch.tensor([[1.0,2.0,3.0]])
b = m(a)

print(a)
print(b)
