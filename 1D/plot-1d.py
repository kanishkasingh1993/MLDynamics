import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pickle
from FourierGridMethod import Fourier_Grid,normalize_wfn,side_wp
from simulation import potential, random_perturbation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.integrate as integrate
import os
cwd = os.getcwd()
def normalize(psi, x, uniform = True):
    int_psi_square = integrate.simps(psi, x)
    return psi/int_psi_square

def save_fig(xmin, xmax, x, D, D_true, V, dt=0.024189*2):
    idx = [0, 6, 9, 26, 90, 149]
    fs = 20
    ts = 25
    xLimits = [xmin, xmax]
    yLimits = [0, 0.16]
    yLimits2= [0, 0.01]
    dLimits = [0, 120]
    for n, i in enumerate(idx):
        fig, ax = plt.subplots(2, figsize=(10, 10), tight_layout=True, gridspec_kw={'height_ratios': [5, 0.5]}, sharex=True)
        ax[0].set_xlim(xLimits)
        ax[0].set_ylim(yLimits)
        ax[1].set_ylim(yLimits2)
        ax[0].grid()
        ax[1].grid()
        ax2 = ax[0].twinx()
        ax[0].set_xlim(xLimits)
        ax2.set_ylim(dLimits)
        ax[1].set_xlabel('$\phi_d$ (degree)', fontsize = fs)
        ax[0].set_ylabel('$|\psi|^2$', fontsize = fs)
        ax[1].set_ylabel('$|\Delta|\psi|^2|$', fontsize = fs)
        ax2.set_ylabel("$V$ (meV)", fontsize = fs)
        plt.yticks(fontsize=ts)
        ax[1].tick_params(axis='x', labelsize=ts)
        ax[1].tick_params(axis='y', labelsize=ts)    
        ax[0].tick_params(axis='y', labelsize=ts)
        line, = ax[0].plot(x, D[i], 'r-', lw=2)
        lineS2, = ax2.plot(x,V[i], 'k-', label="potential", lw=2)
        lineS3, = ax[0].plot(x,D_true[i], 'b-', label="Split-operator", lw=2)
        time_text = ax[0].text(0.1, 0.95, "",ha='left', va='top', transform=ax[0].transAxes, fontsize=25, color='red')
        Diff, = ax[1].plot(x,abs(D[i]-D_true[i]), 'dimgrey', lw=2)
        s = 't = '+ ("%.3f"%(dt*i))+' ps'
        time_text.set_text(s)
        if n == 0:
            plt.subplots_adjust(hspace=0.05)
            Diff.axes.xaxis.set_ticklabels([])         
            lineS2.axes.yaxis.set_ticklabels([])
            ax[1].set_xlabel('', fontsize = fs)
            ax[0].set_ylabel('$|\psi|^2$', fontsize = fs)
            ax[1].set_ylabel('$|\Delta|\psi|^2|$', fontsize = fs)
            ax2.set_ylabel('', fontsize = fs)
        elif n == 1:
            plt.subplots_adjust(hspace=50)
            Diff.axes.xaxis.set_ticklabels([])
            Diff.axes.yaxis.set_ticklabels([])          
            line.axes.yaxis.set_ticklabels([])
            lineS2.axes.yaxis.set_ticklabels([])
            lineS3.axes.yaxis.set_ticklabels([])
            ax[1].set_xlabel('', fontsize = fs)
            ax[0].set_ylabel('', fontsize = fs)
            ax[1].set_ylabel('', fontsize = fs)
            ax2.set_ylabel('', fontsize = fs) 
        elif n == 2:
            plt.subplots_adjust(hspace=10)
            Diff.axes.xaxis.set_ticklabels([])
            Diff.axes.yaxis.set_ticklabels([])          
            lineS3.axes.yaxis.set_ticklabels([])
            line.axes.yaxis.set_ticklabels([])
            ax[1].set_xlabel('', fontsize = fs)
            ax[0].set_ylabel('', fontsize = fs)
            ax[1].set_ylabel('', fontsize = fs)
            ax2.set_ylabel("$V$ (meV)", fontsize = fs) 
        elif n == 3:
            plt.subplots_adjust(hspace=0.05)
            plt.yticks(fontsize=ts)
            ax[1].tick_params(axis='x', labelsize=ts)
            ax[1].tick_params(axis='y', labelsize=ts)    
            ax[0].tick_params(axis='y', labelsize=ts)
            lineS2.axes.yaxis.set_ticklabels([])
            ax[1].set_xlabel('$\phi_d$ (degree)', fontsize = fs)
            ax[0].set_ylabel('$|\psi|^2$', fontsize = fs)
            ax[1].set_ylabel('$|\Delta|\psi|^2|$', fontsize = fs)
            ax2.set_ylabel('', fontsize = fs)   
        elif n == 4:
            plt.subplots_adjust(hspace=10)
            Diff.axes.yaxis.set_ticklabels([])          
            line.axes.yaxis.set_ticklabels([])
            lineS2.axes.yaxis.set_ticklabels([])
            lineS3.axes.yaxis.set_ticklabels([])
            ax[1].set_xlabel('$\phi_d$ (degree)', fontsize = fs)
            ax[0].set_ylabel('', fontsize = fs)
            ax[1].set_ylabel('', fontsize = fs)
            ax2.set_ylabel('', fontsize = fs)   
        elif n == 5:
            plt.subplots_adjust(hspace=10)
            Diff.axes.yaxis.set_ticklabels([])
            line.axes.yaxis.set_ticklabels([])  
            lineS3.axes.yaxis.set_ticklabels([])
            ax[1].set_xlabel('$\phi_d$ (degree)', fontsize = fs)
            ax[0].set_ylabel('', fontsize = fs)
            ax[1].set_ylabel('', fontsize = fs)
            ax2.set_ylabel("$V$ (meV)", fontsize = fs)   
        plt.savefig(cwd+'/picture/MCMC-'+str(i)+'.png')
    return

source_path = cwd+'/result/MCMC-result.pickle'
with open(source_path, 'rb') as handle:
    data_dict = pickle.load(handle)

D=data_dict.get('D')
out=data_dict.get('out')
V_tot_au=data_dict.get('V')

x_min, x_max = -90*np.pi/180, 90*np.pi/180
y_min, y_max = 0, 7.25
x_size, y_size = 128, 300

#Define x and y grids
x_grid = np.linspace(x_min, x_max, x_size)
y_grid = np.linspace(y_min, y_max, y_size)
x_dim, t_dim = np.meshgrid(x_grid, y_grid)
ev_scale=1/(27.211*1e3)
V_pot=potential(x_grid,y_grid)
V = ev_scale*(V_pot)

normal_D = D
normal_out = out
for i in range(len(D)):
    normal_D[i] = normalize(D[i], x_grid*(180/np.pi))
for i in range(len(out)):
    normal_out[i] = normalize(out[i], x_grid*(180/np.pi))

save_fig(x_min*(180/np.pi),x_max*(180/np.pi),x_grid*(180/np.pi),normal_out,normal_D,V/ev_scale, dt=0.0241667*2)