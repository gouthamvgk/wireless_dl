import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torch.nn.functional as F
import random as rn
import matplotlib.pyplot as plt
from noise import GaussianNoise, Noise_1, Noise_2
import os

class comm_4_1(nn.Module):
    def __init__(self, num_sym, num_chan, rate, batch_size = 200, train_snr = 7, hidden_neurons=50):
        super(comm_4_1, self).__init__()
        self.num_symbols = num_sym
        self.num_channels = num_chan
        self.Ebno = 10.0**(train_snr/10.0)  #db eqivalent
        self.std_dev = np.sqrt(1/(2*self.Ebno * rate))
        self.lin1 = nn.Linear(self.num_symbols, self.num_symbols)
        self.lin2 = nn.Linear(self.num_symbols, hidden_neurons)
        self.lin3 = nn.Linear(hidden_neurons, 2)
        self.lin_c = nn.Linear(2, self.num_channels*2)
        self.norm1 = nn.BatchNorm1d(self.num_channels*2)
        self.noise = GaussianNoise((batch_size, self.num_channels * 2), std = self.std_dev)
        self.lin4 = nn.Linear(self.num_channels*2, hidden_neurons)
        self.lin5 = nn.Linear(hidden_neurons, self.num_symbols)
        self.lin6 = nn.Linear(self.num_symbols, self.num_symbols)
        #self.softmax = nn.LogSoftmax(dim=1)
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
        chan_out = self.noise(out)
        rec_out = self.lin4(chan_out)
        chan_out = F.relu(rec_out)
        rec_out = self.lin5(rec_out)
        rec_out = F.relu(rec_out)
        rec_out = self.lin6(rec_out)
        #rec_out = self.softmax(rec_out)

        return rec_out

class comm_4_2(nn.Module):
    def __init__(self, num_sym, num_chan, rate, batch_size = 200, train_snr = 7, hidden_neurons=50):
        super(comm_4_2, self).__init__()
        self.num_symbols = num_sym
        self.num_channels = num_chan
        self.Ebno = 10.0**(train_snr/10.0)  #db eqivalent
        self.std_dev = np.sqrt(1/(2*self.Ebno * rate))
        self.lin1 = nn.Linear(self.num_symbols, self.num_symbols)
        self.lin2 = nn.Linear(self.num_symbols, hidden_neurons)
        self.lin3 = nn.Linear(hidden_neurons, 2)
        self.lin_c = nn.Linear(2, self.num_channels*2)
        self.norm1 = nn.BatchNorm1d(self.num_channels*2)
        self.noise = Noise_1((batch_size, self.num_channels * 2), std = self.std_dev)
        self.lin4 = nn.Linear(self.num_channels*2, hidden_neurons)
        self.lin5 = nn.Linear(hidden_neurons, self.num_symbols)
        self.lin6 = nn.Linear(self.num_symbols, self.num_symbols)
        #self.softmax = nn.LogSoftmax(dim=1)
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
        chan_out = self.noise(out)
        rec_out = self.lin4(chan_out)
        chan_out = F.relu(rec_out)
        rec_out = self.lin5(rec_out)
        rec_out = F.relu(rec_out)
        rec_out = self.lin6(rec_out)
        #rec_out = self.softmax(rec_out)

        return rec_out


class comm_16_1(nn.Module):
    def __init__(self, num_sym, num_chan, rate, batch_size = 200, train_snr = 7, hidden_neurons=100):
        super(comm_16_1, self).__init__()
        self.num_symbols = num_sym
        self.num_channels = num_chan
        self.Ebno = 10.0**(train_snr/10.0)  #db eqivalent
        self.std_dev = np.sqrt(1/(2*self.Ebno * rate))
        self.lin1 = nn.Linear(self.num_symbols, self.num_symbols)
        self.lin2 = nn.Linear(self.num_symbols, hidden_neurons)
        self.lin3 = nn.Linear(hidden_neurons, 2)
        self.lin_c = nn.Linear(2, self.num_channels*2)
        self.norm1 = nn.BatchNorm1d(self.num_channels*2)
        self.noise = GaussianNoise((batch_size, self.num_channels * 2), std = self.std_dev)
        self.lin4 = nn.Linear(self.num_channels*2, hidden_neurons)
        self.lin5 = nn.Linear(hidden_neurons, self.num_symbols)
        self.lin6 = nn.Linear(self.num_symbols, self.num_symbols)
        #self.softmax = nn.LogSoftmax(dim=1)
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
        chan_out = self.noise(out)
        rec_out = self.lin4(chan_out)
        chan_out = F.tanh(rec_out)
        rec_out = self.lin5(rec_out)
        rec_out = F.tanh(rec_out)
        rec_out = self.lin6(rec_out)
        #rec_out = self.softmax(rec_out)

        return rec_out


class comm_16_2(nn.Module):
    def __init__(self, num_sym, num_chan, rate, batch_size = 200, train_snr = 7, hidden_neurons=100):
        super(comm_16_2, self).__init__()
        self.num_symbols = num_sym
        self.num_channels = num_chan
        self.flag = 0
        self.Ebno = 10.0**(train_snr/10.0)  #db eqivalent
        self.std_dev = np.sqrt(1/(2*self.Ebno * rate))
        self.lin1 = nn.Linear(self.num_symbols, self.num_symbols)
        self.lin2 = nn.Linear(self.num_symbols, hidden_neurons)
        self.lin3 = nn.Linear(hidden_neurons, 2)
        self.lin_c = nn.Linear(2, self.num_channels*2)
        self.norm1 = nn.BatchNorm1d(self.num_channels*2)
        self.noise = Noise_2((batch_size, self.num_channels * 2), std = self.std_dev)
        self.lin4 = nn.Linear(self.num_channels*2, hidden_neurons)
        self.lin5 = nn.Linear(hidden_neurons, self.num_symbols)
        self.lin6 = nn.Linear(self.num_symbols, self.num_symbols)
        #self.softmax = nn.LogSoftmax(dim=1)
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
        chan_out = self.noise(out, self.flag%5)
        self.flag += 1
        rec_out = self.lin4(chan_out)
        chan_out = F.tanh(rec_out)
        rec_out = self.lin5(rec_out)
        rec_out = F.tanh(rec_out)
        rec_out = self.lin6(rec_out)
        #rec_out = self.softmax(rec_out)

        return rec_out
