from html.entities import name2codepoint

import gym
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import Linear, ReLU, Dropout, BatchNorm1d
import os
import tqdm

class NN(nn.module):
        def __init__(self):
                super(NN, self).__init__()
                self.network = torch.nn.Sequential(
                        torch.Linear(81+81, 256),
                        torch.ReLU(inplace=True),
                        torch.Linear(256, 256),
                        torch.ReLU(inplace=True),
                        torch.Linear(256, 81),
                        torch.ReLU(inplace=True)
                )



        def forward(self, curr_state, prev_move):
                x = torch.tensor(curr_state+prev_move)
                return self.network(x)
                
