#!/usr/bin/env python
# coding: utf-8

'''
Different machine learning models which targeting PDE solving are writing in here.
Those models are named as following:
fully connected neural network, Deep Operator network (DeepONet), FNO3D and FNO4D.
These models will be tested for solving 1D to 3D time dependent Schrödinger equation under harmonic potential or different potential.

The input of DeepONet is in two parts (x_branch and x_trunk). 
The first part (x_branch) is the initial funciton, which is psi_0 in our case, and the second part (x_trunk) of the input is the domain of the desire function.
There are two DeepONet, the first one which is without CartesianProd means it only predicts point of the function while the second one predicts the funciton in one shot.

FNO3D and FNO4D are written in different style. The 3D model uses convolutional layer while 4D model uses the linear layer. 
Therefore, the input of the 3D model has shape (batch, channel, t_dim, x_dim, y_dim) and the input of the 4D model has shape (batch, t_dim, x_dim, y_dim, channel)

18.12.2023
For DeepONet, I decide all the activations in the model will be GELU. All initializer is fixed as xavier_normal (Glorot normal) for wights and zeros for bias.
'''

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

activation = torch.nn.GELU()
initializer = torch.nn.init.xavier_normal_
initializer_zero = torch.nn.init.zeros_
dtype = torch.float32

class FNN(torch.nn.Module):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes):
        super().__init__()
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i], dtype=dtype))
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)
        self.activation = activation
    def forward(self, inputs):
        x = inputs
        for j, linear in enumerate(self.linears[:-1]):
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
        x = self.linears[-1](x)
        return x
    
class DeepONet(torch.nn.Module):
    """Deep operator network.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
    ):
        super().__init__()
        self.branch = FNN(layer_sizes_branch)
        self.trunk = FNN(layer_sizes_trunk)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = torch.from_numpy(inputs[1])
        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        x_loc = activation(self.trunk(x_loc))
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum("bi,bi->b", x_func, x_loc)
        x = torch.unsqueeze(x, 1)
        # Add bias
        x += self.b
        return x
    
class DeepONetCartesianProd(torch.nn.Module):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk
    ):
        super().__init__()
        self.branch = FNN(layer_sizes_branch)
        self.trunk = FNN(layer_sizes_trunk)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        #x_loc = torch.from_numpy(inputs[1]).float()
        #x_loc = x_loc.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        x_loc = activation(self.trunk(x_loc))
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum("bi,bni->bn", x_func, x_loc)
        # Add bias
        x += self.b
        return x

# Starting from here is for FNO3D and 4D.
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        """
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        mode1 : int
            Number of modes to keep in Fourier Layer (first dimension)
        mode2 : int
            Number of modes to keep in Fourier Layer (second dimension)
        mode3 : int
            Number of modes to keep in Fourier Layer (third dimension)
        """
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        
        self.fft_norm = 'backward'

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.complex64))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.complex64))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.complex64))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.complex64))
        
    def forward(self, x):

        batchsize, channels, height, width, depth = x.shape

        x = torch.fft.rfftn(x.float(), norm=self.fft_norm, dim=[-3, -2, -1])

        out_fft = torch.zeros([batchsize, self.out_channels, height, width, depth//2 + 1], device=x.device, dtype=torch.cfloat)
        
        out_fft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(x[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        
        out_fft[:, :, :self.modes1, -self.modes2:, :self.modes3] = self.compl_mul3d(x[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights2)
        
        out_fft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul3d(x[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights3)
        
        out_fft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul3d(x[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        x = torch.fft.irfftn(out_fft, s=(height, width, depth), norm=self.fft_norm)
        return x
    
    def compl_mul3d(self, a, b):
        '''
        (batch, in_channel, x, y, t), (in_channel, out_channel, x, y, t) -> (batch, out_channel, x, y, t)
        '''
        return torch.einsum("bixyz,ioxyz->boxyz", a, b)

class MLP(nn.Module):
    '''
    Multi-Layer Perceptron
    '''
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FourierBlocks(nn.Module):
    def __init__(self, width, modes1, modes2, modes3):
        super(FourierBlocks, self).__init__()
        self.speconv = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.mlp = MLP(width, width, width)
        self.w = nn.Conv3d(width, width, 1)
        
    def forward(self, x):
        x1 = self.speconv(x)
        x1 = self.mlp(x1)
        x2 = self.w(x)
        x = x1+x2
        x = F.gelu(x)
        return x

class Lifting(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.fc(x)

class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.fc1 = nn.Conv3d(in_channels, hidden_channels, 1)
        self.fc2 = nn.Conv3d(hidden_channels, out_channels, 1)
        self.non_linear = F.gelu
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linear(x)
        x = self.fc2(x)
        return x

class FNO3d(nn.Module):
    def __init__(self, width, modes1, modes2, modes3, lifting_input = 1, output_ch = 1):
        super(FNO3d, self).__init__()
        
        self.lift = Lifting(lifting_input, width)
        #self.lift = nn.Linear(lifting_input, width)
        self.FNOBlock1 = FourierBlocks(width, modes1, modes2, modes3)
        self.FNOBlock2 = FourierBlocks(width, modes1, modes2, modes3)
        self.FNOBlock3 = FourierBlocks(width, modes1, modes2, modes3)
        self.FNOBlock4 = FourierBlocks(width, modes1, modes2, modes3)
        self.project = Projection(width, output_ch, width)
        
    def forward(self, x):
        x = self.lift(x)
        x = self.FNOBlock1(x)
        x = self.FNOBlock2(x)
        x = self.FNOBlock3(x)
        x = self.FNOBlock4(x)
        x = self.project(x)
        
        return x

class SpectralConv4d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, modes4):
        super(SpectralConv4d, self).__init__()

        """
        4D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights5 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights6 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights7 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights8 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul4d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyzt,ioxyzt->boxyzt", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-4,-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-4), x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4], self.weights3)
        out_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes4], self.weights4)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4], self.weights5)
        out_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:, :self.modes4], self.weights6)
        out_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes4], self.weights7)
        out_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:, :self.modes4], self.weights8)        

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-4), x.size(-3), x.size(-2), x.size(-1)))
        return x

class Block4d(nn.Module):
    def __init__(self, width, width2, modes1, modes2, modes3, modes4, out_dim):
        super(Block4d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        
        self.width = width
        self.width2 = width2
        self.out_dim = out_dim
        self.padding = 8
        
        # channel
        self.conv0 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.conv1 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.conv2 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.conv3 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, self.width2)
        self.fc2 = nn.Linear(self.width2, self.out_dim)
        
    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z, size_t = x.shape[2], x.shape[3], x.shape[4], x.shape[5]
        
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_t)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_t)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_t)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_t)
        x = x1 + x2
        
        # pad the domain if input is non-periodic
        x = x[:, :, self.padding:-self.padding, self.padding*2:-self.padding*2, self.padding*2:-self.padding*2, self.padding:-self.padding] 
        
        x = x.permute(0, 2, 3, 4, 5, 1)  
        x1 = self.fc1(x)
        x = F.gelu(x1)
        x = self.fc2(x)
        
        return x
    
class FNO4d(nn.Module):
    def __init__(self, modes1, modes2, modes3, modes4, width, in_dim):
        super(FNO4d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.width = width
        self.width2 = width*4
        self.in_dim = in_dim
        self.out_dim = 1
        self.padding = 8  # pad the domain if input is non-periodic
        
        self.fc0 = nn.Linear(self.in_dim, self.width)
        self.conv = Block4d(self.width, self.width2, 
                               self.modes1, self.modes2, self.modes3, self.modes4, self.out_dim)

    def forward(self, x, gradient=False):
        x = self.fc0(x) 
        x = x.permute(0, 5, 1, 2, 3, 4)
        x = F.pad(x, [self.padding, self.padding, self.padding*2, self.padding*2, self.padding*2, 
                      self.padding*2, self.padding, self.padding])  
        
        x = self.conv(x)
    
        return x
