import torch
from neuraloperator.loss import H1Loss
from helper import print_2D_single, print_1D
import numpy as np
import os
cwd = os.getcwd()
l = H1Loss(d=3)
x = np.linspace(-9,9,65)
T = np.linspace(0,10,51)
pred = torch.load(cwd+'/result/anharmonic_FNO_dens-pred.pt')
true = torch.load(cwd+'/result/anharmonic_FNO_dens-true.pt')
V = torch.load(cwd+'/result/anharmonic_FNO_dens-potential.pt')
psi_t = pred[1][10][0]
psi_true = true[1][10][0]
potential = V[1][10]
for t in [5,10,25,50]:
    path = cwd+'/picture/2D-an-compare-'+str(t)+'.png'
    print_2D_single(psi_true, psi_t, path, potential, p='an', idx = t)
    diff = abs(psi_t[t] - psi_true[t])
    print(diff.max())
    q = diff.argmax()//65
    path = cwd+'/picture/1D-an-compare-'+str(t)+'.png'
    ground = psi_true[t][q]
    predict = psi_t[t][q]
    V = potential[q]
    title = 'at t = '+str(T[t])+' a.u., $q_1 = $' + str(x[q])+ ' a.u.'
    print_1D(ground,predict,path,V,title)
