#!/usr/bin/env python
# coding: utf-8

'''
Here contains four classes for FNO, SpectralConv2d, MLP, FNOBlocks, FNO2d. (and one more for Projection)
SpectralConv2d and MLP are used in FNOBlocks, and FNOBlocks is used FNO2d.

The structure of FNO2d is 5 layers (1+4+1).
Input (x) is first lifted up by using Conv2d. Then there are 4 FNOBlocks.
Each FNOBlock is bulided by a SpectralConv2d and MLP with a Conv2d, then a non-linear function.
There are 4 FNOBlock.
Then it is projected back to the physical space by two Conv2d and a non-linear function.

A class called DataLoader2D, and two classes for the loss funciton LpLoss and H1Loss.
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DataLoader2D(object):
    def __init__(self, u0, uout, V, dt, scaling, frames1=0, frames2=300):
        self.u0=u0
        self.uout=uout
        self.dt=dt
        self.V=V
        self.scaling = scaling
        self.frames1 = frames1
        self.frames2 = frames2
        
    def make_loader(self, gridx, gridt, ntrain, batch_size, start=0, train=True):
        gridx=torch.from_numpy(gridx)
        gridt=torch.from_numpy(gridt)
        
        Vet=self.V[start:start+ntrain]
        u0 = self.u0[start:start+ntrain]
        Ys=self.uout[start:start+ntrain]
        
        u0_real = u0.abs()
        u0_real = u0_real**2
        V_perturbed = Vet*self.scaling
        
        nt=len(gridt)
        nx=len(gridx)
        
        u0_real = u0_real.reshape(ntrain, 1, nx)
        u0_real = u0_real.repeat([1, nt,1])

        gridx = gridx.reshape(1, 1, nx)
        gridt = gridt.reshape(1, nt, 1)
        
        x1=gridx.repeat([ntrain, nt, 1])
        t1=gridt.repeat([ntrain, 1, nx])
                
        Xs = torch.stack([V_perturbed, u0_real, t1, x1], dim=1)
        
        Xs = Xs[:,:,self.frames1:self.frames2,:]
        Ys = Ys[:,:,self.frames1:self.frames2,:]
        Vet = Vet[:,self.frames1:self.frames2,:]
        
        dataset = torch.utils.data.TensorDataset(Xs, Vet, Ys)
        
        if train:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            print('input shape: ', Xs.shape)
            print('output shape: ', Ys.shape)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

#Set fix{x,y,z}_bnd if function is non-periodic in {x,y,z} direction
#x: (*, s)
#y: (*, s)
def central_diff_1d(x, h, fix_x_bnd=False):
    dx = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h)

    if fix_x_bnd:
        dx[...,0] = (x[...,1] - x[...,0])/h
        dx[...,-1] = (x[...,-1] - x[...,-2])/h
    
    return dx

#x: (*, s1, s2)
#y: (*, s1, s2)
def central_diff_2d(x, h, fix_x_bnd=True, fix_y_bnd=True):
    if isinstance(h, float):
        h = [h, h]

    dx = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))/(2.0*h[0])
    dy = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h[1])

    if fix_x_bnd:
        dx[...,0,:] = (x[...,1,:] - x[...,0,:])/h[0]
        dx[...,-1,:] = (x[...,-1,:] - x[...,-2,:])/h[0]
    
    if fix_y_bnd:
        dy[...,:,0] = (x[...,:,1] - x[...,:,0])/h[1]
        dy[...,:,-1] = (x[...,:,-1] - x[...,:,-2])/h[1]
        
    return dx, dy

#x: (*, s1, s2, s3)
#y: (*, s1, s2, s3)
def central_diff_3d(x, h, fix_x_bnd=True, fix_y_bnd=True, fix_z_bnd=True):
    if isinstance(h, float):
        h = [h, h, h]

    dx = (torch.roll(x, -1, dims=-3) - torch.roll(x, 1, dims=-3))/(2.0*h[0])
    dy = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))/(2.0*h[1])
    dz = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h[2])

    if fix_x_bnd:
        dx[...,0,:,:] = (x[...,1,:,:] - x[...,0,:,:])/h[0]
        dx[...,-1,:,:] = (x[...,-1,:,:] - x[...,-2,:,:])/h[0]
    
    if fix_y_bnd:
        dy[...,:,0,:] = (x[...,:,1,:] - x[...,:,0,:])/h[1]
        dy[...,:,-1,:] = (x[...,:,-1,:] - x[...,:,-2,:])/h[1]
    
    if fix_z_bnd:
        dz[...,:,:,0] = (x[...,:,:,1] - x[...,:,:,0])/h[2]
        dz[...,:,:,-1] = (x[...,:,:,-1] - x[...,:,:,-2])/h[2]
        
    return dx, dy, dz


class H1Loss(object):
    def __init__(self, d=1, L=2*math.pi, reduce_dims=0, reductions='sum', fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
        super().__init__()

        assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3-D."

        self.d = d
        self.fix_x_bnd = fix_x_bnd
        self.fix_y_bnd = fix_y_bnd
        self.fix_z_bnd = fix_z_bnd

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims
        
        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L
    
    def compute_terms(self, x, y, h):
        dict_x = {}
        dict_y = {}

        if self.d == 1:
            dict_x[0] = x
            dict_y[0] = y

            x_x = central_diff_1d(x, h[0], fix_x_bnd=self.fix_x_bnd)
            y_x = central_diff_1d(y, h[0], fix_x_bnd=self.fix_x_bnd)

            dict_x[1] = x_x
            dict_y[1] = y_x
        
        elif self.d == 2:
            dict_x[0] = torch.flatten(x, start_dim=-2)
            dict_y[0] = torch.flatten(y, start_dim=-2)

            x_x, x_y = central_diff_2d(x, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)
            y_x, y_y = central_diff_2d(y, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-2)
            dict_x[2] = torch.flatten(x_y, start_dim=-2)

            dict_y[1] = torch.flatten(y_x, start_dim=-2)
            dict_y[2] = torch.flatten(y_y, start_dim=-2)
        
        else:
            dict_x[0] = torch.flatten(x, start_dim=-3)
            dict_y[0] = torch.flatten(y, start_dim=-3)

            x_x, x_y, x_z = central_diff_3d(x, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)
            y_x, y_y, y_z = central_diff_3d(y, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-3)
            dict_x[2] = torch.flatten(x_y, start_dim=-3)
            dict_x[3] = torch.flatten(x_z, start_dim=-3)

            dict_y[1] = torch.flatten(y_x, start_dim=-3)
            dict_y[2] = torch.flatten(y_y, start_dim=-3)
            dict_y[3] = torch.flatten(y_z, start_dim=-3)
        
        return dict_x, dict_y

    def uniform_h(self, x):
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)
        
        return h
    
    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        
        return x
        
    def abs(self, x, y, h=None):
        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
            
        dict_x, dict_y = self.compute_terms(x, y, h)

        const = math.prod(h)
        diff = const*torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += const*torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
        
        diff = diff**0.5

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff
        
    def rel(self, x, y, h=None):
        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
        
        dict_x, dict_y = self.compute_terms(x, y, h)

        diff = torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2
        ynorm = torch.norm(dict_y[0], p=2, dim=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
            ynorm += torch.norm(dict_y[j], p=2, dim=-1, keepdim=False)**2
        
        diff = (diff**0.5)/(ynorm**0.5)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff


    def __call__(self, x, y, h=None):
        return self.rel(x, y, h=h)

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
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        
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

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.fft_norm = 'backward'
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, a, b):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x):
        batchsize, channels, height, width = x.shape
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x.float(), norm=self.fft_norm)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros([batchsize, self.out_channels, height, width//2 + 1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(height, width), dim=(-2,-1), norm=self.fft_norm)
        return x

    
class MLP(nn.Module):
    '''
    Multi-Layer Perceptron
    '''
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FourierBlocks(nn.Module):
    def __init__(self, width, modes1, modes2):
        super(FourierBlocks, self).__init__()
        self.speconv = SpectralConv2d(width, width, modes1, modes2)
        self.mlp = MLP(width, width, width)
        self.w = nn.Conv2d(width, width, 1)
        
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
        self.fc = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.fc(x)

class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1)
        self.non_linear = F.gelu
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linear(x)
        x = self.fc2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, width, modes1, modes2, lifting_input = 4, output_channel = 1):
        super(FNO2d, self).__init__()
        
        self.output_channel = output_channel
        self.lifting_input = lifting_input
        self.lift = Lifting(lifting_input, width)
        self.FNOBlock1 = FourierBlocks(width, modes1, modes2)
        self.FNOBlock2 = FourierBlocks(width, modes1, modes2)
        self.FNOBlock3 = FourierBlocks(width, modes1, modes2)
        self.FNOBlock4 = FourierBlocks(width, modes1, modes2)
        self.project = Projection(width, output_channel, width)
        
    def forward(self, x):
        if self.lifting_input != 4:
            x = x[:,1:,:,:]
        x = self.lift(x)
        x = self.FNOBlock1(x)
        x = self.FNOBlock2(x)
        x = self.FNOBlock3(x)
        x = self.FNOBlock4(x)
        x = self.project(x)
        
        return x
