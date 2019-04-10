import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torch.nn.functional as F
import random as rn
import matplotlib.pyplot as plt
import os


device = "cuda" if torch.cuda.is_available() else "cpu"

class GaussianNoise(nn.Module):
    def __init__(self, size, std, mean = 0):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std
        self.value = torch.zeros(size).to(device)

    def forward(self, inp):
        temp = self.value.data.normal_(0, std = self.std)
        return inp + temp


class Noise_1(nn.Module):
    def __init__(self, size, std, mean = 0):
        super(Noise_1, self).__init__()
        self.mean = mean
        self.std = std
        self.value = torch.zeros(size).to(device)

    def forward(self, inp):
        temp = self.value.data.normal_(0, std = self.std)
        temp_size = inp.size()
        save_array = inp.detach().cpu().numpy()
        save_array.tofile('/home/gou/np')
        os.system('python2 /home/gou/pro.py')
        result = np.fromfile('/home/gou/np1', dtype=np.float32).reshape(temp_size)
        result = result - save_array
        had_noise = torch.from_numpy(result).to(device)
        return inp + temp + had_noise


class Noise_2(nn.Module):
    def __init__(self, size, std, mean = 0):
        super(Noise_2, self).__init__()
        self.mean = mean
        self.std = std
        self.value = torch.zeros(size).to(device)
        self.hyperparameters = {0:[-100,0.8,0.1,0.1,0.1,-0.2],
                                1:[-120,1.0,0.3,0.3,0.4, -0.4],
                                2:[-130,1.0,0.3,0.3,0.4,-0.5],
                                3:[-140,1.0,0.3,0.3,0.5,-0.6],
                                4:[-150,1.0,0.4,0.4,0.5,-0.7]}

    def forward(self, inp, hyp_index):
        temp = self.value.data.normal_(0, std = self.std)
        temp_size = inp.size()
        save_array = inp.detach().cpu().numpy()
        save_array.tofile('/home/gou/np')
        command = 'python2 /home/gou/pro.py --pn_mag {} --iq_mag_imb {} --iq_ph_imb {} --quad_offset {} --inp_offset \
{} --freq_offset {}'.format(*self.hyperparameters[hyp_index])
        #print(command)
        os.system(command)
        result = np.fromfile('/home/gou/np1', dtype=np.float32).reshape(temp_size)
        result = result - save_array
        had_noise = torch.from_numpy(result).to(device)
        return inp + temp + had_noise
