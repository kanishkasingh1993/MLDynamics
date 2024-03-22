import numpy as np
import torch
import matplotlib.pyplot as plt
from helper import setup_FNO, dataloader_FNO, print_LC, print_2D, train_FNO, test_FNO
from neuraloperator.loss import H1Loss

random_seed = 1997
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

#setup parameters
name = 'anharmonic_FNO_dens'
dim = 3
ntrain = 0.2
width = 40
modes1 = modes2 = modes3 = 20
epochs = 500
batch_size = 20
potential_in = True
random = True
save_potential = True
lr = 0.009
step = 50
gamma = 0.99
data_path = '/home/leek97/ML/data/data-2D/'+name+'.pt'
lc_path = '/home/leek97/ML/picture/' + name + '-LC.png'
compare_path = '/home/leek97/ML/picture/' + name + '-compare.png'
result_path = '/home/leek97/ML/result/' + name
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#load data
train_loader, test_loader, ntrain, ntest, in_channel, nt, nx, ny = dataloader_FNO(data_path, ntrain, batch_size, potential = potential_in, random = random)
#setup model
model, optimizer, scheduler = setup_FNO(device, width, modes1, modes2, modes3, in_channel, lr, step, gamma)
#setup loss function
lf = H1Loss(d=dim)

#train the model
model, training_loss = train_FNO(model, ntrain, train_loader, optimizer, scheduler, device, epochs, lf)

#print the learning curve
loss = np.array(training_loss)
loss_h1 = loss[:,0]
loss_mse = loss[:,1]
print_LC(loss_h1, lc_path)

#evulation and print the result
pred_out, true_out = test_FNO(model, ntest, test_loader, device, lf, result_path, save_potential=save_potential)
psi_t = pred_out[0][10]
psi_true = true_out[0][10]
for i, (x, y) in enumerate(test_loader):
    if i == 0:
        V = x[10,-1,0,:,:]
    else:
        break
print_2D(psi_true, psi_t, compare_path, potential = V)
#print the memory left in device 0
free, total = torch.cuda.mem_get_info()
print('Free memory: ', free/1000000000, 'GB')
print('Total memory: ', total/1000000000, 'GB')