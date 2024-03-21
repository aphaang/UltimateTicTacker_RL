import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import os
import tqdm

class NN(nn.Module):
        def __init__(self, input_dims, fc1_dims, fc2_dims, output_dims=1, lr=0.001):
                super(NN, self).__init__()
                self.input_dims = input_dims
                self.fc1_dims = fc1_dims
                self.fc2_dims = fc2_dims
                self.output_dims = output_dims
                self.lr = lr
                self.nn = nn.Sequential(
                        nn.Linear(input_dims, fc1_dims),
                        nn.ReLU(inplace=True),
                        nn.Linear(fc1_dims, fc2_dims),
                        nn.ReLU(inplace=True),
                        nn.Linear(fc2_dims, output_dims)
                )
                self.optimizer = Adam(self.nn.parameters(), lr=self.lr)
                self.loss = torch.nn.MSELoss()
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.to(self.device)



        def forward(self, input):
                x = torch.tensor(input.float())
                return self.nn(x)
                
        def save(self, name):
                torch.save(self,os.path.dirname(__file__)+"/DQNetwork" + name + ".pt")

        def load(self, name):
                self = torch.load(os.path.dirname(__file__)+"/DQNetwork" + name + ".pt")