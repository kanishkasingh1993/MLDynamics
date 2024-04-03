#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import torch.nn.functional as F
import numpy as np
from bayesian import pop_i
from scipy import signal
from math import pi
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class split_op() :
    def __init__(self,xmin,xmax,Nx,nt,dt,t_end,sig=0.1,amp=10.0, device=None,dtype=torch.float64):
        #Basic initialisations
        self.xmin=xmin
        self.xmax=xmax
        self.Nx=Nx
        
        #define x grid
        x=torch.linspace(xmin,xmax,Nx,device=device, dtype=dtype)
        self.x=x
        self.dt=dt
        self.t_end=t_end
        self.device=device
        self.dx=(xmax-xmin)/Nx
        #define k grid
        self.dk = pi / xmax
        self.amp=amp
        self.k = torch.cat((torch.arange(0, Nx / 2 ),
                                torch.arange(-Nx / 2, 0)),0) * self.dk
        #self.k=torch.linspace(xmin,xmax,Nx,device=device, dtype=dtype)*self.dk
        
        #define operator grids
        self.V = torch.empty(Nx, dtype=torch.cfloat)
        self.R = torch.empty(Nx, dtype=torch.cfloat)
        self.K = torch.empty(Nx, dtype=torch.cfloat)
        #self.wfc0 = np.empty(Nx, dtype=torch.cfloat)
        
        self.u = torch.zeros_like(x, device=device,dtype=torch.cfloat)
        self.u0 = torch.zeros_like(self.u, device=device,dtype=torch.cfloat)
        self.sig = sig
        self.t = 0
        self.it = 0
        self.U = []
        self.T = []
        self.D = []
        self.im_time=False
        
        #operators
        self.wfcoffset=0.5
        self.timesteps=nt
         
        
    def setup(self,V_set,i):
        Irel=1542050
        #Irel=1.0
        if self.im_time:
            K = torch.exp(-0.5*(1/Irel) *(self.k ** 2) * self.dt)
            R = torch.exp(-0.5 * V_set[i] * self.dt)
        else:
            K = torch.exp(-0.5 *(1/Irel)* (self.k ** 2) * self.dt * 1j)
            R = torch.exp(-0.5 * V_set[i] * self.dt * 1j)
        return K,R

    def split_op(self,u0,V_set) :
      
        self.T = []
        self.U = []
        #print(V_set)
      
        self.u=u0
        self.u0=u0  
        #u0=self.u
        #self.U.append(u0)
        for i in range(self.timesteps):
           
            self.K,self.R=self.setup(V_set,i)
            #print (self.K,self.R)
            # Half-step in real space
            self.u *= self.R

            # FFT to momentum space
            self.u = torch.fft.fft(self.u)

            # Full step in momentum space
            self.u *= self.K

            # iFFT back
            self.u = torch.fft.ifft(self.u)

            # Final half-step in real space
            self.u *= self.R

            # Density for plotting and potential
            density = (self.u.abs()) ** 2

            # Renormalizing for imaginary time
            #if self.im_time:
            #  renorm_factor = sum(density) * self.dx
            #  self.u /= sqrt(renorm_factor)
            self.U.append(self.u)
            self.D.append(density)
        
        return torch.stack(self.U),torch.stack(self.D),u0

def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def random_perturbation (x_size,y_size,coh_x,coh_y,amp_var):
    # Define the grid dimensions
    # Generate a 2D white noise signal
    noise = np.random.randn(y_size, x_size)
    
    # Generate the Gaussian filter kernels
    filt_x = signal.gaussian(x_size, std=amp_var*coh_x/(np.sqrt(np.pi)))
    filt_y = signal.gaussian(y_size, std=amp_var*coh_y/(np.sqrt(np.pi)))
    filt = np.outer(filt_y, filt_x)

    # Apply the filter to the noise signal
    filtered_noise = signal.convolve(noise, filt, mode='same')

    # Scale the amplitudes to match the desired variance
    std_noise = np.std(filtered_noise)
    scaled_noise = (filtered_noise / std_noise) * np.sqrt(amp_var)
    
    return scaled_noise

def potential (x_array,y_array):
    """
    Double well potential in meV.
    """
    
    X,Y=np.meshgrid(x_array,y_array)
    return 41.4478-13.5702*np.cos(2*X)+38.1985*np.cos(4*X)+7.365*np.cos(6*X)+0.9870*np.cos(8*X)+0.3590*np.cos(10*X)+0.0813*np.cos(12*X)

def chirped_pulse(t, A0, t0, sigma, f0, beta, phi):
    """
    Generates a chirped pulse with the given parameters at the specified time points. (in a.u.)

    Parameters:
    -----------
    t : array_like
        Time points at which to evaluate the pulse, in atomic units.
    A0 : float
        Peak amplitude of the pulse.
    t0 : float
        Center time of the pulse, in atomic units.
    sigma : float
        Standard deviation of the Gaussian envelope, in atomic units.
    f0 : float
        Central frequency of the pulse, in atomic units.
    beta : float
        Chirp rate, in atomic units.
    phi : float
        Phase offset of the pulse, in radians.

    Returns:
    --------
    pulse : ndarray
        The generated pulse at the specified time points.
    """
    pulse = A0 * np.exp(-0.5 * ((t - t0) / sigma) ** 2) * np.cos(2 * np.pi * (f0 + beta * (t - t0)) * (t - t0) + phi)

    return pulse

def alpha(phi):
    """
    polarizability in a.u.
    """
    alpha_xx=16.832*np.cos(phi)+200.871
    alpha_yy=-20.1435*np.cos(phi)+112.4955
    alpha_xy=--18.81*np.sin(phi)
    
    alpha_0=alpha_xx*(np.cos(phi))**2+alpha_yy*(np.sin(phi))**2-2*alpha_xy*np.sin(phi)*np.cos(phi)
    
    return alpha_0

def complete_simulation(x_grid, t_grid, pot_params, params=None, timestep=1000, MCMC=False, pulse = True, overlap=True):
    
    """
    Complete simulation which returns the population score
    """

    x_size=len(x_grid)
    y_size=len(t_grid)
    x_max=np.max(x_grid)
    x_min=np.min(x_grid)
    x_dim, t_dim = np.meshgrid(x_grid, t_grid)
    
    #pot_params={'psi0':psi0,'psit':psit,'V_vib':V_vib,'alpha':alpha_1, 'I_rel': Irel}
    psi_start=pot_params.get('psi_start',1)
    psi_ref=pot_params.get('psi_ref',2)
    V_vib_au=pot_params.get('V_vib',3)
    alpha_1=pot_params.get('alpha',4)
    Irel=pot_params.get('I_rel',5)
    
    phi = np.pi                   # Phase offset #fixed

    #Propagate the wavepacket
    if pulse:
        if params is None:
            params = {'sig':1,'t0':0,'beta':0,'A0':0.0238,'f0':1}
        sig = params.get('sig', 1.0)
        t0 = params.get('t0', 2.0)
        beta = params.get('beta', 3.0)
        A0 = params.get('A0', 4.0)
        f0 = params.get('f0', 5.0)
    
        #chirped_pulse(y_grid, A0, t0, sigma, f0, beta, phi)
        elec_f = chirped_pulse(t_dim, A0, t0, sig, f0, beta, phi)
        V_tot_au=V_vib_au-0.25*(elec_f**2)*(alpha_1)
    else:
        V_tot_au=V_vib_au
    
    rabi=split_op(xmin=x_min,xmax=x_max,Nx=x_size,nt=y_size,dt=timestep,t_end=timestep*y_size, device=None, dtype=torch.float64)

    #Set the wavepacket to an intital wavepacket
    u = torch.tensor(psi_start,dtype=torch.complex64)
    
    #Split_operator
    U,D,ux=rabi.split_op(u,torch.tensor(V_tot_au))
    
    #plot_wavepacket(x_grid,V_vib[0],psi_sim)
    
    if not MCMC:
        to_out=(U,D,ux,V_tot_au)
    else:
        psi_sim=U[y_size-1].numpy()
    
        if len(psi_start.shape) > 1:
            pop_score=np.zeros(len(psi_ref[0,:]))
            for i in range(len(psi_ref[0,:])):
                pop_score[i]= pop_i(x_grid,psi_ref[:,i],psi_sim)
            
            to_out=np.sum(pop_score)
        else:
            if overlap:
                to_out = pop_i(x_grid,psi_ref,psi_sim)
            else:
                psi_ref = torch.from_numpy(psi_ref.copy())
                psi_sim = torch.from_numpy(psi_sim)
                to_out = 1-F.mse_loss((psi_ref.abs())**2,(psi_sim.abs())**2)
    return to_out
