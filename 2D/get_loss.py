import torch
from neuraloperator.loss import H1Loss
from helper import print_2D
import os
cwd = os.getcwd()
l = H1Loss(d=3)
pred = torch.load(cwd+'/result/anharmonic_FNO_dens-pred.pt')
true = torch.load(cwd+'/result/anharmonic_FNO_dens-true.pt')
V = torch.load(cwd+'/result/anharmonic_FNO_dens-potential.pt')
path = cwd+'/picture/2D-an-compare.png'
psi_t = pred[1][10][0]
psi_true = true[1][10][0]
potential = V[1][10]
print_2D(psi_true, psi_t, path, potential, p='an')
print('H1 Loss = ', l(psi_t, psi_true))
