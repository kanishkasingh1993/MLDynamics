#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import numpy as np
from Generate_fields import random_perturbation
from new_split import split_op
from bayesian import pop_i

# In[ ]:


def potential (x_array,y_array):
    X,Y=np.meshgrid(x_array,y_array)
    return 41.4478-13.5702*np.cos(2*X)+38.1985*np.cos(4*X)+7.365*np.cos(6*X)+ \
+0.9870*np.cos(8*X)+0.3590*np.cos(10*X)+0.0813*np.cos(12*X)


# In[ ]:


def electric_field(k,k1,t0,t1,sigma,x,t):
    
    
    #return   A / (1 + np.exp(-k*(t-t0))) - A / (1 + np.exp(k*t0))
    #return A*np.exp(-(t - t0)**2 / (2 * k**2))*np.cos((x-x0)*np.pi/180)
    #np.exp(-(x - x0)**2 / (2 * k1**2))
    #return A*(np.sin((t-t0)*np.pi/180))**2*np.exp(-(x - x0)**2 / (2 * k1**2))
    tmax=90*np.pi/180
    f1 = np.where(t <= t0, t/t0, 1)
    f2 = 0.2*np.where(t > t0, (tmax - t)/(tmax - t0), 0)
    A=k*f1 * f2
    

    
    
    #A=k*np.abs(t-t0)
    phi=k1*(t-t0)
    #np.exp(-(t - t0)**2 / (2*sigma**2))
    #



    return  0.8*A*np.exp(-(t - t1)**2 / (2*sigma**2))*(np.sin(phi))**2



def chirped_pulse(t, A0, t0, sigma, f0, beta, phi):
    """
    Generates a chirped pulse with the given parameters at the specified time points.

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
    #each timestep needs to be mutliplied by 990 which is the simulation timestep
    t=990*t
    pulse = A0 * np.exp(-0.5 * ((t - t0) / sigma) ** 2) \
            * np.cos(2 * np.pi * (f0 + beta * (t - t0)) * (t - t0)/sigma + phi)

    return pulse



# In[ ]:


def alpha(phi):
    
    #phi=phi*np.pi/180
    
    alpha_xx=16.832*np.cos(phi)+200.871
    alpha_yy=-20.1435*np.cos(phi)+112.4955
    alpha_xy=--18.81*np.sin(phi)
    
    alpha_0=alpha_xx*(np.cos(phi))**2+alpha_yy*(np.sin(phi))**2-2*alpha_xy*np.sin(phi)*np.cos(phi)
    
    return alpha_0

def complete_simulation(task,x_grid,t_grid,pot_params,params=None):
    
    """
    Complete simulation which returns the population score
    """
    
    
    if params is None:
        params = {'sig':0.8,'t0':2.0,'t1':3,'k':8,'k1':10}
    sig = params.get('sig', 1.0)
    t0 = params.get('t0', 2.0)
    t1 = params.get('t1', 3.0)
    k = params.get('k', 4.0)
    k1 = params.get('k1', 5.0)
    
    x_size=len(x_grid)
    y_size=len(t_grid)
    
    x_max=np.max(x_grid)
    x_min=np.min(x_grid)
    
    x_dim, t_dim = np.meshgrid(x_grid, t_grid)
    
    #rng = np.random.RandomState(seed=123)

    #pot_params={'psi0':psi0,'psit':psit,'V_vib':V_vib,'alpha':alpha_1, 'I_rel': Irel}
    psi_start=pot_params.get('psi_start',1)
    psi_ref=pot_params.get('psi_ref',2)
    V_vib=pot_params.get('V_vib',3)
    alpha_1=pot_params.get('alpha',4)
    Irel=pot_params.get('I_rel',5)

    ev_scale=1/(27*1e3)

    #coherence and variance in amplitude for term2 this is in ev unless scaled
    eta_coh_x = 12.2
    eta_coh_y = 0.27
    
    #Generate electric field
    #elec_f = electric_field(k,k1,t0,t1,sig, x_dim, t_dim)
    
    A0 = 1                    # Peak amplitude #Variable range (1,5)
    t0 = 3*990                      # Center time #fixed
    sigma = t0               # Standard deviation #variable (0.5,2)
    f0 = 0                      # Central frequency #variable (2,5)
    beta =0.0005                # Chirp rate fixed
    phi = 1.51*np.pi  
    #phi = np.pi                   # Phase offset #fixed
    
    #chirped_pulse(y_grid, A0, t0, sigma, f0, beta, phi)

    elec_f = chirped_pulse(t_dim, k, t0, sigma, f0, beta, phi)

    #elec_f = chirped_pulse(k,k1,t0,t1,sig, x_dim, t_dim)
    
    # Lets assume that V_vib is in eV, then first it needs to be 
    
    A1=np.amax(elec_f)
    print(A1)
    
    eta_amp_var=11.9*4/(A1)**2 #this is in meV
    #convert to au by multiplying by 1/27*1e3
    eta_amp_au=eta_amp_var*ev_scale
    #Generate the n_alpha term and plot it 
    eta_perturb=random_perturbation(x_size,y_size,eta_coh_x,eta_coh_y,eta_amp_au)


    #Propagate the wavepacket
    V_tot=V_vib-0.25*(elec_f**2)*(alpha_1+eta_perturb)
    #V_tot_au=ev_scale*(V_tot)
    
    timestep=990
    rabi=split_op(xmin=x_min,xmax=x_max,Nx=x_size,nt=y_size,dt=timestep,t_end=timestep*y_size, device=None,dtype=torch.float64)

    
    
    #Set the wavepacket to an intital wavepacket
    u=torch.tensor(psi_start,dtype=torch.complex64)

    #Split_operator
    U,D,ux=rabi.split_op(u,torch.tensor(V_tot))
    
    psi_sim=U[y_size-1].numpy()
    
    pop_score=np.zeros(len(psi_ref[0,:]))
    for i in range(len(psi_ref[0,:])):
        pop_score[i]= pop_i(x_grid,psi_ref[:,i],psi_sim)
        
    
    #plot_wavepacket(x_grid,V_vib[0],psi_sim)
    
    if task=='gen':
        to_out=(U,D,ux,V_tot)
    else:
        to_out=np.sum(pop_score)
    return to_out