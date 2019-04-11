# WIRELESS SYSTEM USING NEURAL NETWORK

This repo contains the code for my final year project which is to implement a wireless communication system using neural network instead of traditional algorithms.

## CONCEPT
Communication algorithms in practice use a mathematical model which is derived from mathematical analysis.  These systems are very complex and requires a lot of knowledge about the channel.  Here a instead of the traditional algorithms a two sided neural network structure called Autoencoder is implemented to replace it.
The Autoencoder has two neural networks for transmitter and receiver.  They are trained jointly with channel layer in the middle where different type of noises are added.

### DEPENDENCIES

 - Python 3
 - GNU radio
 - Pytorch
 - Numpy
 - Matplotlib

### TRAINING
Two kind of model is implemented.  One for 4 symbols and the other one for 16 symbols. Training is carried out in two different steps. In the 1st step only simple gaussian noise is added and the model is trained to learn a robust representation for it.
In the second step more complex hardware impairment noise and channel noise are added and the model is finetuned on the previous weights.
The following commands can be run accordingly to see the results,

 - `python 4sym_step1.py` - To train 1st step of 4 symbol system.
 - `python 4sym_step2.py` - To train 2nd step of 4 symbol system.
 - `python 16sym_step1.py`- To train 1st step of 16 symbol system.
 - `python 16sym_step1.py` - To train 2nd step of 16 symbol system.

All the above commands can be run with following command line arguments

 - --num_channels  -> No of channels for transmission
 - --no_epochs -> No of epochs for training
 - --batch_size -> Batch size to be used for training
 - --lr -> Learning rate for gradient updates
 - --hidden_neurons -> No of neurons in the hidden layers

### REFERENCES
1. [Sebastian Dörner, Sebastian Cammerer, Jakob Hoydis and Stephan ten Brink (2018) ‘Deep Learning Based Communication Over the Air](https://arxiv.org/pdf/1707.03384.pdf)’

2. [Tim O’Shea and Jakob Hoydis (2017) ‘An Introduction to Deep Learning for the Physical Layer’](https://arxiv.org/pdf/1702.00832.pdf)

3. [Timothy J. O'Shea, Kiran Karra and T. Charles Clancy (2017) ‘Learning to communicate: Channel auto-encoders, domain specific regularizers, and attention](https://arxiv.org/pdf/1608.06409.pdf)
