import torch
import numpy as np
from neuraloperator.loss import H1Loss
from helper import print_2D, henon_heiles, renormalize
l = H1Loss(d=3)
pred = torch.load('/home/leek97/ML/result/hh_FNO_dens-1-pred_out.pt').reshape(100,51,65,65)
true = torch.load('/home/leek97/ML/result/hh_FNO_dens-1-true_out.pt').reshape(100,51,65,65)
path = '/home/leek97/ML/picture/FNO-bounded.png'
X, Y = np.meshgrid(np.linspace(-9,9,65), np.linspace(-9,9,65))
potential = henon_heiles(X,Y,0.111803)
psi_t = pred[0]
#for i in range(psi_t.shape[0]):
#    if psi_t.min() < 0:
#        psi_t[i] = psi_t[i]-psi_t.min()

#for i in range(psi_t.shape[0]):
#    psi_t[i] = renormalize(psi_t[i])
psi_true = true[0]
print_2D(psi_true, psi_t, path, potential, p='hh')
print('H1 Loss = ', l(psi_t, psi_true))

pred = torch.load('/home/leek97/ML/result/hh_DON_dens-1-pred_out.pt').reshape(100,51,65,65)
true = torch.load('/home/leek97/ML/result/hh_DON_dens-1-true_out.pt').reshape(100,51,65,65)
path = '/home/leek97/ML/picture/DON-bounded.png'
psi_t = pred[0]
psi_true = true[0]
print_2D(psi_true, psi_t, path, potential, p='hh')
print('H1 Loss = ', l(psi_t, psi_true))

pred = torch.load('/home/leek97/ML/result/anharmonic_FNO_dens-0-pred_out.pt')
true = torch.load('/home/leek97/ML/result/anharmonic_FNO_dens-0-true_out.pt')
V = torch.load('/home/leek97/ML/result/anharmonic_FNO_dens-0-V.pt')
path = '/home/leek97/ML/picture/FNO-an-high.png'
psi_t = pred[0][10]
psi_true = true[0][10]
potential = V[0][10]
print_2D(psi_true, psi_t, path, potential, p='an')
print('H1 Loss = ', l(psi_t, psi_true))
