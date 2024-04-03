'''
These helpers are for 2D data only. Both DeepONet and FNO are here.
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from neuraloperator.model import DeepONetCartesianProd, FNO3d
from neuraloperator.loss import H1Loss
from sklearn.utils import shuffle
import scipy
def henon_heiles(x,y,l):
    return 0.5*(x**2)+0.5*(y**2)+l*((x**2)*y-(1/3)*y**3)

def setup_DON(m, n_dim, width, n_layers, device, lr = 0.001, step = 100, gamma = 0.9, DDP = True, details = True):
    print('Setting up DeepONet.')
    x_branch_layers = [m]
    x_trunk_layers = [n_dim]
    for i in range(n_layers):
        x_branch_layers += [width]
        x_trunk_layers += [width]
        model = DeepONetCartesianProd(x_branch_layers, x_trunk_layers)
    if DDP:
        model = torch.nn.DataParallel(model)
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        model = model.to(device)
    if details:
        print("Model architecture")
        print(model)
        print()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of trainable parameters: ', params)
        print()

    print('Setting up optimizer and scheduler.')
    print('Learing rate: ', lr)
    print('scheduler gamma: ', gamma)
    print('scheduler step: ', step)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

    return model, optimizer, scheduler

def setup_FNO(device, width, modes1, modes2, modes3, in_channel=4, lr = 0.001, step = 100, gamma = 0.9, wf = False, details = True):
    print('Setting up FNO.')
    if wf:
        model = FNO3d(width=width, modes1=modes1, modes2=modes2, modes3=modes3, lifting_input = in_channel, output_ch = 2)
    else:
        model = FNO3d(width=width, modes1=modes1, modes2=modes2, modes3=modes3, lifting_input = in_channel)
    model = model.to(device)
    if details:
        print("Model architecture")
        print(model)
        print()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of trainable parameters: ', params)
        print()

    print('Setting up optimizer and scheduler.')
    print('Learing rate: ', lr)
    print('scheduler gamma: ', gamma)
    print('scheduler step: ', step)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

    return model, optimizer, scheduler
    
def dataloader_DON(datapath, ntrain, ntest=None, batch_size=20, potential = False, random=False):
    data_all = torch.load(datapath)
    x_branch = data_all['x_branch']
    size = x_branch.shape
    ntotal = size[0]
    ntrain = int(ntrain*ntotal)
    if not ntest:
        ntest = ntotal - ntrain
    else:
        ntest = int(ntest)
    m = np.prod(size[1:])
    x_branch = x_branch.reshape(ntotal, m)
    if not potential:
        x_trunk = data_all['x_trunk'][:,:,:3]
    else:
        x_trunk = data_all['x_trunk']
    y = data_all['y']
    if random:
        x_branch, x_trunk, y = shuffle(x_branch, x_trunk, y, random_state=0)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_branch[:ntrain], x_trunk[:ntrain], y[:ntrain]),
        batch_size = batch_size,
        shuffle = True
    )

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_branch[-ntest:], x_trunk[-ntest:], y[-ntest:]),
        batch_size = batch_size,
        shuffle = False
    )
    print('number of training data: ', ntrain)
    for i, (x, y, z) in enumerate(train_loader):
        if i == 0:
            print('Input function shape: ', x.shape)
            print('Input location shape: ', y.shape)
            print('Output data shape: ', z.shape)
        else:
            break
    return train_loader, test_loader, ntrain, ntest

def dataloader_FNO(datapath, ntrain, ntest=None, batch_size=20, potential = False, random=False):
    data_all = torch.load(datapath)
    psi0 = data_all['x_conv']
    y = data_all['y']
    if not potential:
        psi0 = psi0[:,:4,...]
    if type(y) == np.ndarray:
        y = torch.from_numpy(y)
    ntotal = psi0.shape[0]
    in_channel = psi0.shape[1]
    ntrain = int(ntotal*ntrain)
    if not ntest:
        ntest = ntotal - ntrain
    else:
        ntest
    if random:
        psi0, y = shuffle(psi0, y, random_state=0)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(psi0[:ntrain], y[:ntrain]),
        batch_size = batch_size,
        shuffle = True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(psi0[-ntest:], y[-ntest:]),
        batch_size = batch_size,
        shuffle = False
    )
    for i, (x, y) in enumerate(train_loader):
        if i == 0:
            print('Input function shape: ', x.shape)
            print('Output data shape: ', y.shape)
            _, _, nt, nx, ny = x.shape
        else:
            break
    return train_loader, test_loader, ntrain, ntest, in_channel, nt, nx, ny
    
def print_LC(train_loss, path):
    epochs = len(train_loss)

    plt.close()
    plt.plot(np.linspace(1,epochs,epochs), train_loss, label='h1')
    plt.title('Loss versus epoches')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('epoches')
    plt.savefig(path)
    print('Learning curve figure saved at ', path)

    return

def print_2D(psi_true, psi_t, path, potential, p = 'an', idx = [5,10,25,50], fix=True):
    if type(psi_true) == list:
        psi_true = np.array(psi_true)
    elif type(psi_true) == torch.Tensor:
        psi_true = psi_true.cpu()

    if type(psi_t) == list:
        psi_t = np.array(psi_t)
    elif type(psi_t) == torch.Tensor:
        psi_t = psi_t.cpu()
    t = np.linspace(0,10,51)
    X, Y = np.meshgrid(np.linspace(-9,9,65), np.linspace(-9,9,65))
    V_pot = potential
    if p == 'hh':
        level = np.linspace(0, np.sqrt(120), 10)
    else:
        level = np.linspace(0, np.sqrt(800), 10)
    level = level**2
    mycmap = plt.get_cmap('hot_r')
    mycmap2 = plt.get_cmap('hot')
    line = 1.0
    style = 'dashed'
    fs = 12
    fig, axs = plt.subplots(3, 4, figsize=(8,6), sharex=True, sharey=True, layout='constrained')
    plt.subplots_adjust(left=0.08,
                        bottom=0.1,
                        right=0.95,
                        top=0.95,
                        wspace=0.15,
                        hspace=0.1)
    axs[0][0].contour(X, Y, V_pot, level, cmap=mycmap,linestyles=style,linewidths=line)
    axs[0][1].contour(X, Y, V_pot, level, cmap=mycmap,linestyles=style,linewidths=line)
    axs[0][2].contour(X, Y, V_pot, level, cmap=mycmap,linestyles=style,linewidths=line)
    axs[0][3].contour(X, Y, V_pot, level, cmap=mycmap,linestyles=style,linewidths=line)
    axs[1][0].contour(X, Y, V_pot, level, cmap=mycmap,linestyles=style,linewidths=line)
    axs[1][1].contour(X, Y, V_pot, level, cmap=mycmap,linestyles=style,linewidths=line)
    axs[1][2].contour(X, Y, V_pot, level, cmap=mycmap,linestyles=style,linewidths=line)
    axs[1][3].contour(X, Y, V_pot, level, cmap=mycmap,linestyles=style,linewidths=line)
    psi1 = axs[0][0].pcolormesh(X,Y,(psi_true)[idx[0]],shading='auto', cmap=mycmap2)
    psi2 = axs[0][1].pcolormesh(X,Y,(psi_true)[idx[1]],shading='auto', cmap=mycmap2)
    psi5 = axs[0][2].pcolormesh(X,Y,(psi_true)[idx[2]],shading='auto', cmap=mycmap2)
    psi10 = axs[0][3].pcolormesh(X,Y,(psi_true)[idx[3]],shading='auto', cmap=mycmap2)
    psi_pred_1 = axs[1][0].pcolormesh(X,Y,(psi_t)[idx[0]],shading='auto', cmap=mycmap2)  
    psi_pred_2 = axs[1][1].pcolormesh(X,Y,(psi_t)[idx[1]],shading='auto', cmap=mycmap2)
    psi_pred_5 = axs[1][2].pcolormesh(X,Y,(psi_t)[idx[2]],shading='auto', cmap=mycmap2)
    psi_pred_10 = axs[1][3].pcolormesh(X,Y,(psi_t)[idx[3]],shading='auto', cmap=mycmap2)
    psi_diff_1 = axs[2][0].pcolormesh(X,Y,(abs(psi_true-psi_t))[idx[0]],shading='auto')
    psi_diff_2 = axs[2][1].pcolormesh(X,Y,(abs(psi_true-psi_t))[idx[1]],shading='auto')
    psi_diff_5 = axs[2][2].pcolormesh(X,Y,(abs(psi_true-psi_t))[idx[2]],shading='auto')
    psi_diff_7 = axs[2][3].pcolormesh(X,Y,(abs(psi_true-psi_t))[idx[3]],shading='auto')
    if fix:
        psi1.set_clim(0,0.35)
        psi2.set_clim(0,0.35)
        psi5.set_clim(0,0.35)
        psi10.set_clim(0,0.35)
        psi_pred_1.set_clim(-0.005,0.35)
        psi_pred_2.set_clim(-0.005,0.35)
        psi_pred_5.set_clim(-0.005,0.35)
        psi_pred_10.set_clim(-0.005,0.35)
        psi_diff_1.set_clim(0,0.0405)
        psi_diff_2.set_clim(0,0.0405)
        psi_diff_5.set_clim(0,0.0405)
        psi_diff_7.set_clim(0,0.0405)
    axs[0][0].set_title('t = '+str(t[idx[0]])+' a.u.')
    axs[0][1].set_title('t = '+str(t[idx[1]])+' a.u.')
    axs[0][2].set_title('t = '+str(t[idx[2]])+' a.u.')
    axs[0][3].set_title('t = '+str(t[idx[3]])+' a.u.')
    cb1 = fig.colorbar(psi10)
    cb2 = fig.colorbar(psi_pred_10)
    cb3 = fig.colorbar(psi_diff_7)
    cb1.set_label('$|\psi|^2$', fontsize = fs)
    cb2.set_label('$|\psi|^2$', fontsize = fs)
    cb3.set_label('$|\Delta|\psi|^2|$', fontsize = fs)

    plt.setp(axs[-1, :], xlabel='$q_2$ (a.u.)')
    plt.setp(axs[:, 0], ylabel='$q_1$ (a.u.)')
    plt.savefig(path)
    print('Comparsion figure saved at ', path)

    return

def print_2D_single(psi_true, psi_t, path, potential, p = 'an', idx = 5):
    if type(psi_true) == list:
        psi_true = np.array(psi_true)
    elif type(psi_true) == torch.Tensor:
        psi_true = psi_true.cpu()

    if type(psi_t) == list:
        psi_t = np.array(psi_t)
    elif type(psi_t) == torch.Tensor:
        psi_t = psi_t.cpu()
    t = np.linspace(0,10,51)
    X, Y = np.meshgrid(np.linspace(-9,9,65), np.linspace(-9,9,65))
    V_pot = potential
    if p == 'hh':
        level = np.linspace(0, np.sqrt(120), 10)
    else:
        level = np.linspace(0, np.sqrt(800), 10)
    level = level**2
    mycmap = plt.get_cmap('hot_r')
    mycmap2 = plt.get_cmap('hot')
    line = 1.0
    style = 'dashed'
    fs = 12
    fig, axs = plt.subplots(3, 1, figsize=(3,6), sharex=True, sharey=True, layout='constrained')
    plt.subplots_adjust(left=0.08,
                        bottom=0.1,
                        right=0.95,
                        top=0.95,
                        wspace=0.15,
                        hspace=0.1)
    axs[0].contour(X, Y, V_pot, level, cmap=mycmap,linestyles=style,linewidths=line)
    axs[1].contour(X, Y, V_pot, level, cmap=mycmap,linestyles=style,linewidths=line)
    psi = axs[0].pcolormesh(X,Y,(psi_true)[idx],shading='auto', cmap=mycmap2)
    psi_pred = axs[1].pcolormesh(X,Y,(psi_t)[idx],shading='auto', cmap=mycmap2)  
    psi_diff = axs[2].pcolormesh(X,Y,(abs(psi_true-psi_t))[idx],shading='auto')
    if idx == 50:
        psi_diff.set_clim(0,0.0405)
    elif idx == 25:
        psi_diff.set_clim(0,0.0206)
    elif idx == 10:
        psi_diff.set_clim(0,0.03)
    axs[0].set_title('t = '+str(t[idx])+' a.u.')
    cb1 = fig.colorbar(psi)
    cb2 = fig.colorbar(psi_pred)
    cb3 = fig.colorbar(psi_diff)
    cb1.set_label('$|\psi|^2$', fontsize = fs)
    cb2.set_label('$|\psi|^2$', fontsize = fs)
    cb3.set_label('$|\Delta|\psi|^2|$', fontsize = fs)
    if idx == 25:
        #xticks = cb3.get_ticks()
        #cb3.set_ticklabels(['0.00', '','0.01', '', '0.02'])
        cb3.ax.locator_params(nbins=3)
    plt.setp(axs[-1], xlabel='$q_2$ (a.u.)')
    plt.setp(axs, ylabel='$q_1$ (a.u.)')
    plt.savefig(path)
    print('Comparsion figure saved at ', path)
    return

def print_1D(psi_true, psi_t, path, potential, title):
    x = np.linspace(-9,9,65)
    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    ax.grid()
    ax2 = ax.twinx()
    ax.set_xlabel('$q_2$ (a.u.)')
    ax.set_ylabel('$|\psi|^2$')
    ax2.set_ylabel("$V (a.u.)$")
    ax.set_title(title)
    line, = ax.plot(x, psi_t, 'r-', lw=2)
    lineS2, = ax2.plot(x, potential, 'k-', lw=2)
    lineS3, = ax.plot(x, psi_true, 'b-', lw=2)
    plt.savefig(path)
    return

def train_DON(model, ntrain, train_loader, optimizer, scheduler, device, epoches, lf, save_path = False):
    training_loss = []
    print('Training model...')
    print(f'{"Epochs":16}{"train_h1"}{" ":10}{"train_mse"}')
    for e in range(epoches):
        train_h1 = 0
        train_mse = 0
        model.train()
        for x, x_loc, y in train_loader:
            x = x.to(device).float()
            x_loc = x_loc.to(device).float()
            y = y.to(device).float()
            optimizer.zero_grad()
            out = model((x, x_loc))
            out = out.reshape(y.shape)
            h1 = lf(out, y).mean()
            h1.backward()
            mse = torch.nn.functional.mse_loss(out, y, reduction='mean')
            optimizer.step()
            scheduler.step()
            train_h1+=h1.item()
            train_mse+=mse.item()
        train_h1/=ntrain
        train_mse/=ntrain
        if (e%100==0) or e == epoches-1:
            print(f'{str(e):16}{train_h1:.8f}{" ":8}{train_mse:.8f}')
        training_loss.append([train_h1, train_mse])
    if save_path:
        torch.save(model, save_path)
    return model, training_loss

def test_DON(model, ntest, test_loader, device, lf, save_path = False, save_potential=False):
    pred_out = []
    true_out = []
    model.eval()
    test_h1 = 0.0
    test_mse = 0.0
    print()
    print('Testing...')
    with torch.no_grad():
        for x, x_loc, y in test_loader:
            x = x.to(device).float()
            x_loc = x_loc.to(device).float()
            y = y.to(device).float()
            out = model((x, x_loc))
            out = out.reshape(y.shape)
            h1 = lf(out, y).mean()
            mse = torch.nn.functional.mse_loss(out, y, reduction='mean')
            test_h1+=h1.item()
            test_mse+=mse.item()
            pred_out.append((out.cpu()).tolist())
            true_out.append((y.cpu()).tolist())
        test_h1/=ntest
        test_mse/=ntest
        
        print('Done with prediction')
        print('H1Loss: ', test_h1)
        print('mse: ', test_mse)
        print()
    if save_path:
        torch.save(torch.from_numpy(np.array((pred_out))), save_path+'-pred_out.pt')
        torch.save(torch.from_numpy(np.array((true_out))), save_path+'-true_out.pt')
        if save_potential:
            V = []
            for x, x_loc, y in test_loader:
                p = x_loc[:,:,-1].reshape(y.shape)
                p = p[:,0,:,:]
                V.append((p.cpu()).tolist())
                torch.save(torch.from_numpy(np.array(V)), save_path+'-V.pt')
    return pred_out, true_out

def train_FNO(model, ntrain, train_loader, optimizer, scheduler, device, epoches, lf, save_path = False):
    training_loss = []
    print('Training model...')
    print(f'{"Epochs":16}{"train_h1"}{" ":10}{"train_mse"}')
    for e in range(epoches):
        train_h1 = 0
        train_mse = 0
        model.train()
        for x, y in train_loader:
            x = x.to(device).float()
            y = y.to(device).float()
            optimizer.zero_grad()
            out = model(x)
            out = out.reshape(y.shape)
            h1 = lf(out, y).mean()
            h1.backward()
            mse = torch.nn.functional.mse_loss(out, y, reduction='mean')
            optimizer.step()
            scheduler.step()
            train_h1+=h1.item()
            train_mse+=mse.item()
        train_h1/=ntrain
        train_mse/=ntrain
        if (e%100==0) or e == epoches-1:
            print(f'{str(e):16}{train_h1:.8f}{" ":8}{train_mse:.8f}')
        training_loss.append([train_h1, train_mse])
    if save_path:
        torch.save(model, save_path)
    return model, training_loss

def test_FNO(model, ntest, test_loader, device, lf, save_path = False, save_potential=False):
    pred_out = []
    true_out = []
    model.eval()
    test_h1 = 0.0
    test_mse = 0.0
    print()
    print('Testing...')
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device).float()
            y = y.to(device).float()
            out = model(x)
            out = out.reshape(y.shape)
            h1 = lf(out, y).mean()
            mse = torch.nn.functional.mse_loss(out, y, reduction='mean')
            test_h1+=h1.item()
            test_mse+=mse.item()
            pred_out.append((out.cpu()).tolist())
            true_out.append((y.cpu()).tolist())
        test_h1/=ntest
        test_mse/=ntest
        
        print('Done with prediction')
        print('H1Loss: ', test_h1)
        print('mse: ', test_mse)
        print()
    if save_path:
        torch.save(torch.from_numpy(np.array(pred_out)), save_path+'-pred_out.pt')
        torch.save(torch.from_numpy(np.array(true_out)), save_path+'-true_out.pt')
        if save_potential:
            V = []
            for x, y in test_loader:
                V.append((x[:,-1,0,:,:].cpu()).tolist())
                torch.save(torch.from_numpy(np.array(V)), save_path+'-V.pt')

    return pred_out, true_out

def plot_his(means, stds, labels, ylabel_name, xlabel, fig_name, log=False):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel(ylabel_name)
    if log:
        ax.set_yscale('log')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

def henon_heiles(x,y,l):
    return 0.5*(x**2)+0.5*(y**2)+l*((x**2)*y-(1/3)*y**3)
def harmonic(x,y):
    return 0.5*(x**2)+0.5*(y**2)
X, Y = np.meshgrid(np.linspace(-9,9,65), np.linspace(-9,9,65))
ha = harmonic(X,Y)
hh = henon_heiles(X,Y, 0.111803)
picture_path = '/home/leek97/ML/picture/'
nt, nx, ny = 51, 65, 65

def error_mean_std(task, data_path, pic_path, ntest = 100, loss=H1Loss(d=3), details=True, renorm = False):
    if task == 'an':
        V = torch.load(data_path+'-V.pt').reshape(ntest,nx,ny)
    elif task == 'ha':
        V = ha
    elif task == 'hh':
        V = hh
    pred = torch.load(data_path+'-pred_out.pt').reshape(ntest,51,65,65)
    if renorm:
        print('renormalizing the result')
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                pred[i][j] = renormalize(abs(pred[i][j]))
    true = torch.load(data_path+'-true_out.pt').reshape(ntest,51,65,65)
    l=np.array([])
    for i in range(len(pred)):
        l = np.append(l, loss(pred[i], true[i]))
    mean, std = np.mean(l), np.std(l)
    max_idx = np.argmax(l)
    min_idx = np.argmin(l)    
    if details:
        print('Mean loss: ', mean)
        print('STD loss: ', std)
        print('Maximum loss: ', max_idx, l[max_idx])
        print('Minimum loss: ', min_idx, l[min_idx])
    print_2D(true[max_idx], pred[max_idx], pic_path+'-max.png', potential=V[max_idx] if task == 'an' else V, p=task)
    print_2D(true[min_idx], pred[min_idx], pic_path+'-min.png', potential=V[min_idx] if task == 'an' else V, p=task)
    print()
    return mean, std

def renormalize(psi):
    '''
    renormalize a 2D density function.
    return torch.tensor
    '''
    if type(psi) == torch.Tensor:
        psi = psi.numpy()
    norm = np.trapz(np.trapz(psi, axis=0), axis=0)
    psi_normalized = psi / norm
    return torch.from_numpy(psi_normalized)

def check_norm(psi):
    '''
    integrate the inner product of the wavefunction to check the normalization
    return: float
    '''
    if type(psi) == torch.Tensor:
        psi = psi.numpy()
    return np.trapz(np.trapz(psi, axis=0), axis=0)