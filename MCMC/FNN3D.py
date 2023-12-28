#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional
import math
#from .lowrank2d import LowRank2d
#from .basics import SpectralConv2d


# In[2]:


def compl_mul3d(a, b):
    return torch.einsum("bixyz,ioxyz->boxyz", a, b)

def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", a, b)


# In[3]:


def compl_mul2d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    #print(a.shape,b.shape)
    #print(a,b)
    return torch.einsum("bixy,ioxy->boxy", a, b)


# In[4]:


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2,modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,self.modes3, dtype=torch.cfloat))
        
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, activation='tanh'):
        super(FourierBlock, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.speconv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)
        self.linear = nn.Conv1d(in_channels, out_channels, 1)
        if activation == 'tanh':
            self.activation = torch.tanh_
        elif activation == 'gelu':
            self.activation = nn.GELU
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        '''
        input x: (batchsize, channel width, x_grid, y_grid, t_grid)
        '''
        x1 = self.speconv(x)
        x2 = self.linear(x.view(x.shape[0], self.in_channel, -1))
        out = x1 + x2.view(x.shape[0], self.out_channel, x.shape[2], x.shape[3], x.shape[4])
        if self.activation is not None:
            out = self.activation(out)
        return out
        

    


# In[5]:


class FNN3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width=16, fc_dim=128, layers=None, in_dim=4, out_dim=1, activation='tanh', pad_x=0, pad_y=0, pad_z=0):
        '''
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            in_dim: int, input dimension
            out_dim: int, output dimension
        '''
        super(FNN3d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.padding = (0, 0, 0, pad_z, 0, pad_y, 0, pad_x)

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(self.in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv3d(
            in_size, out_size, mode1_num, mode2_num, mode3_num)
            for in_size, out_size, mode1_num, mode2_num, mode3_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, self.out_dim)
        
        if activation =='tanh':
            self.activation = F.tanh
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation == F.relu
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        '''
        Args:
            x: (batchsize, x_grid, y_grid, t_grid, 3)
        Returns:
            u: (batchsize, x_grid, y_grid, t_grid, 1)
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        nx, ny, nz = x.shape[1], x.shape[2], x.shape[3]
        x = F.pad(x, self.padding, "constant", 0)
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y, size_z)
            x = x1 + x2
            if i != length - 1:
                # x = torch.tanh(x)
                x = self.activation(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        # x = torch.tanh(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.reshape(batchsize, size_x, size_y, size_z, self.out_dim) # make sure dimensions are what we expect before getting rid of padding
        x = x[..., :nx, :ny, :nz, :]
        return x


# In[6]:


# width=64
# in_size=4
# out_size=2
# modes1=[15, 12, 9, 9]
# modes2=[15, 12, 9, 9]
# layers = [width] * 4

# sp_convs = nn.ModuleList([SpectralConv2d(
#             in_size, out_size, mode1_num, mode2_num)
#             for in_size, out_size, mode1_num, mode2_num
          
#             in zip(layers, layers[1:], modes1, modes2)])


# In[7]:


#print(list(zip(layers, layers[1:], modes1, modes2)))
#print(layers)
#print(sp_convs)


# In[8]:


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
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


# In[9]:


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)



# In[10]:


class Adam(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            adam(params_with_grad,
                 grads,
                 exp_avgs,
                 exp_avg_sqs,
                 max_exp_avg_sqs,
                 state_steps,
                 amsgrad=group['amsgrad'],
                 beta1=beta1,
                 beta2=beta2,
                 lr=group['lr'],
                 weight_decay=group['weight_decay'],
                 eps=group['eps'])
        return loss

