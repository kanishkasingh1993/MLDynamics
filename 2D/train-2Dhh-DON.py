import numpy as np
import torch
import matplotlib.pyplot as plt
from helper import setup_DON, dataloader_DON, print_LC, print_2D, train_DON, test_DON
from neuraloperator.loss import H1Loss

random_seed = 19526
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

#setup the potential
def henon_heiles(x,y,l):
    return 0.5*(x**2)+0.5*(y**2)+l*((x**2)*y-(1/3)*y**3)
X, Y = np.meshgrid(np.linspace(-9,9,65), np.linspace(-9,9,65))
V = henon_heiles(X,Y, 0.111835)

#setup parameters
name = 'hh_DON_dens'
nt, nx, ny = 51, 65, 65
m = nx*ny
dim = n_dim = 3
ntrain = 0.4
width = 128
n_layers = 5
epochs = 2000
batch_size = 20
potential_in = False
DDP = True
lr = 0.0009
step = 100
gamma = 0.99
data_path = '/home/leek97/ML/data/data-2D/'+name+'.pt'
lc_path = '/home/leek97/ML/picture/' + name + '-LC-1.png'
compare_path = '/home/leek97/ML/picture/' + name + '-compare-1.png'
result_path = '/home/leek97/ML/result/' + name + '-1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load data
train_loader, test_loader, ntrain, ntest = dataloader_DON(data_path, ntrain, ntest = 100, batch_size=batch_size, potential = potential_in)
#setup model
model, optimizer, scheduler = setup_DON(m, n_dim, width, n_layers, device, lr, step, gamma, DDP)
#setup loss function
lf = H1Loss(d=dim)

#train the model
model, training_loss = train_DON(model, ntrain, train_loader, optimizer, scheduler, device, epochs, lf)

#print the learning curve
loss = np.array(training_loss)
loss_h1 = loss[:,0]
loss_mse = loss[:,1]
print_LC(loss_h1, lc_path)

#evulation and print the result
pred_out, true_out = test_DON(model, ntest, test_loader, device, lf, result_path)
psi_t = pred_out[0][0]
psi_true = true_out[0][0]
print_2D(psi_true, psi_t, compare_path, potential = V)
#print the memory lef tin device 0
free, total = torch.cuda.mem_get_info()
print('Free memory: ', free/1000000000, 'GB')
print('Total memory: ', total/1000000000, 'GB')