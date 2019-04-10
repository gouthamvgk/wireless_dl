import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torch.nn.functional as F
import random as rn
import matplotlib.pyplot as plt
from noise import GaussianNoise
import os
import system

from system import comm_16_1
from transmitter import transmitter_16
from receiver import receiver_16
device = "cuda" if torch.cuda.is_available() else "cpu"

domain = [4,16,64]
const_range = {4:1, 16:3, 64:7}
N_symbols = 16
if N_symbols not in domain: raise ValueError('Not the correct number of symbols')
bits = np.log2(N_symbols)
bits = int(bits)
print('No. of possible symbols are {}. Each symbol requires {} bits'.format(N_symbols, bits))
const = const_range[N_symbols]
print('Constellation range is {} -> {}'.format(const, -const))
hidden_neurons = 100

N = 500000
num_channels = 2
rate = bits/num_channels

no_epochs =200
batch_size = 100

com_system = comm(N_symbols, num_channels, rate, batch_size)
com_system = com_system.to(device)
optimizer = optim.Adam(com_system.parameters(), lr = 0.0001)
criterion = nn.CrossEntropyLoss()

trans = transmitter(N_symbols, num_channels)
trans = trans.to(device)
recv = receiver(N_symbols, num_channels)
recv = recv.to(device)
