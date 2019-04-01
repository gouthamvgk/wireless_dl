import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torch.nn.functional as F
import random as rn
import matplotlib.pyplot as plt
import os


class transmitter_4(nn.Module):
    def __init__(self, num_sym, num_chan):
        super(transmitter, self).__init__()
        self.num_symbols = num_sym
        self.num_channels = num_chan
        self.lin1 = nn.Linear(self.num_symbols, self.num_symbols)
        self.lin2 = nn.Linear(self.num_symbols, hidden_neurons)
        self.lin3 = nn.Linear(hidden_neurons, 2)
        self.lin_c = nn.Linear(2, self.num_channels*2)
        self.norm1 = nn.BatchNorm1d(self.num_channels*2)

    def load_state(self, lin1, lin2,lin3,lin_c, norm1):
        self.lin1.load_state_dict(lin1.state_dict())
        self.lin2.load_state_dict(lin2.state_dict())
        self.lin3.load_state_dict(lin3.state_dict())
        self.lin_c.load_state_dict(lin_c.state_dict())
        self.norm1.load_state_dict(norm1.state_dict())

    def forward(self, inp):
        out = self.lin1(inp)
        out = F.relu(out)
        out = self.lin2(out)
        out = F.relu(out)
        out = self.lin3(out)
        out = F.tanh(out)
        out = self.lin_c(out)
        out = self.norm1(out)
        out = F.tanh(out)

        return out


class transmitter_16(nn.Module):
    def __init__(self, num_sym, num_chan):
        super(transmitter, self).__init__()
        self.num_symbols = num_sym
        self.num_channels = num_chan
        self.lin1 = nn.Linear(self.num_symbols, self.num_symbols)
        self.lin2 = nn.Linear(self.num_symbols, hidden_neurons)
        self.lin3 = nn.Linear(hidden_neurons, 2)
        self.lin_c = nn.Linear(2, self.num_channels*2)
        self.norm1 = nn.BatchNorm1d(self.num_channels*2)

    def load_state(self, lin1, lin2,lin3,lin_c, norm1):
        self.lin1.load_state_dict(lin1.state_dict())
        self.lin2.load_state_dict(lin2.state_dict())
        self.lin3.load_state_dict(lin3.state_dict())
        self.lin_c.load_state_dict(lin_c.state_dict())
        self.norm1.load_state_dict(norm1.state_dict())

    def forward(self, inp):
        out = self.lin1(inp)
        out = F.tanh(out)
        out = self.lin2(out)
        out = F.tanh(out)
        out = self.lin3(out)
        out = F.tanh(out)
        out = self.lin_c(out)
        out = self.norm1(out)
        out = F.tanh(out)

        return out
