#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np


# In[ ]:


class DataLoader2D(object):
    def __init__(self,u0,uout,V,dt):

        self.u0=u0
        self.uout=uout
        self.dt=dt
        self.V=V
    def make_loader(self, gridx,gridt,ntrain, batch_size, start=0, train=True):
        
      
        gridx=torch.from_numpy(gridx)
        gridt=torch.from_numpy(gridt)
    
        self.V=self.V[0:ntrain]
        self.u0 = self.u0[0:ntrain]
        Ys=self.uout[0:ntrain]
        u0_real=self.u0.real
        u0_im=self.u0.imag
        Vet=self.V
        #print(u0_real.shape)
        nt=len(gridt)
        nx=len(gridx)
        
        u0_real = u0_real.reshape(ntrain, 1, nx)
        u0_im = u0_im.reshape(ntrain, 1, nx)
        #print(u0_real.shape,u0_im.shape)
        u0_real=u0_real.repeat([1, nt,1])
        u0_im=u0_im.repeat([1, nt,1])
        print(u0_real.shape,Vet.shape)
    
        gridx = gridx.reshape(1, 1, nx)
        gridt = gridt.reshape(1, nt, 1)
        
        x1=gridx.repeat([ntrain, nt, 1])
        t1=gridt.repeat([ntrain, 1, nx])
        #print(self.V.shape)
        
        print(u0_real.shape,Vet.shape,gridx.repeat([ntrain, nt, 1]).shape)
        Xs = torch.stack([u0_real,Vet,gridx.repeat([ntrain, nt, 1]), gridt.repeat([ntrain, 1, nx])], dim=-1)
        dataset = torch.utils.data.TensorDataset(Xs,Vet, Ys)
        print(Xs.shape,Ys.shape,Vet.shape)

        if train:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader

