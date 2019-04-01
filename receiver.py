import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torch.nn.functional as F
import random as rn
import matplotlib.pyplot as plt
import os

class receiver_4(nn.Module):
    def __init__(self, num_sym, num_chan):
        super(receiver, self).__init__()
        self.num_symbols = num_sym
        self.num_channels = num_chan
        self.lin3 = nn.Linear(self.num_channels*2, hidden_neurons)
        self.lin4 = nn.Linear(hidden_neurons, self.num_symbols)
        self.lin5 = nn.Linear(self.num_symbols, self.num_symbols)
        self.softmax = nn.Softmax(dim=1)

    def load_state(self, lin3, lin4, lin5):
        self.lin3.load_state_dict(lin3.state_dict())
        self.lin4.load_state_dict(lin4.state_dict())
        self.lin5.load_state_dict(lin5.state_dict())

    def forward(self, inp):
        rec_out = self.lin3(inp)
        chan_out = F.relu(rec_out)
        rec_out = self.lin4(rec_out)
        rec_out = F.relu(rec_out)
        rec_out = self.lin5(rec_out)
        rec_out = self.softmax(rec_out)

        return rec_out


class receiver_16(nn.Module):
    def __init__(self, num_sym, num_chan):
        super(receiver, self).__init__()
        self.num_symbols = num_sym
        self.num_channels = num_chan
        self.lin3 = nn.Linear(self.num_channels*2, hidden_neurons)
        self.lin4 = nn.Linear(hidden_neurons, self.num_symbols)
        self.lin5 = nn.Linear(self.num_symbols, self.num_symbols)
        self.softmax = nn.Softmax(dim=1)

    def load_state(self, lin3, lin4, lin5):
        self.lin3.load_state_dict(lin3.state_dict())
        self.lin4.load_state_dict(lin4.state_dict())
        self.lin5.load_state_dict(lin5.state_dict())

    def forward(self, inp):
        rec_out = self.lin3(inp)
        chan_out = F.tanh(rec_out)
        rec_out = self.lin4(rec_out)
        rec_out = F.tanh(rec_out)
        rec_out = self.lin5(rec_out)
        rec_out = self.softmax(rec_out)

        return rec_out
