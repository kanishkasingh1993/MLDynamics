#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.linalg import eigh
from scipy.integrate import simps

# In[4]:


def Fourier_Grid(Irel,N,L,V_sampled):
    
    """
    Calculate the wavefunction and eigenenergies of the Hamiltonian H(φd) = -(1/2Irel)∂^2/∂φd^2 + V_sampled(φd)
    using the Approximate Fourier Grid Hamiltonian method.

    Parameters:
    - V_sampled (array): array of potential energy values sampled at discrete points in φd space
    - Irel (float): relative moment of inertia
    - N (int): number of grid points in φd space
    - L (float): length of the φd space

    Returns:
    - E (array): array of eigenenergies in ascending order
    - psi (array): array of wavefunctions corresponding to the eigenenergies
    """    
    pi=np.pi
    K=pi/(L/N)
    print((1/Irel)*(0.5/3)*(K**2))
    H=np.zeros([N,N])
    
    for i in range(N):
    
        for j in range(i+1):
            if i ==j :
                H[i,j]=(1/Irel)*(0.5/3)*(K**2) + V_sampled[i]
            else :
                H[i,j]= (1/Irel)*K**2/pi**2 * (-1.)**(j-i)/(j-i)**2
                H[j,i]=H[i,j]
            
            
    E, psi = eigh(H)
    
    return E,psi
    


# In[5]:


def FourierGridHamiltonian(Irel, N, L,V_sampled):
    """
    Calculate the wavefunction and eigenenergies of the Hamiltonian H(φd) = -(1/2Irel)∂^2/∂φd^2 + V_sampled(φd)
    using the Fourier Grid Hamiltonian method.

    Parameters:
    - V_sampled (array): array of potential energy values sampled at discrete points in φd space
    - Irel (float): relative moment of inertia
    - N (int): number of grid points in φd space
    - L (float): length of the φd space

    Returns:
    - eigenvalues (array): array of eigenenergies in ascending order
    - eigenvectors (array): array of wavefunctions corresponding to the eigenenergies
    """

    # Create the grid in φd space
    x = np.linspace(-L/2, L/2, N)
    dx = x[1] - x[0]
    K=np.pi/dx
   


    # Create the kinetic energy matrix
    H = np.zeros((N, N))

    for i in range(N):
        for j in range(i+1):
            if i ==j :
                H[i,j]=(0.5/3)*(1/Irel)*(K**2)*(1+2/N**2) + V_sampled[i]
            else :
                H[i,j]= (1/Irel)*(K**2/N**2)*((-1.)**(j-i))/(np.sin(np.pi*(j-i)/N))**2
                H[j,i]=H[i,j]
        

    # Diagonalize the Hamiltonian matrix
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    return eigenvalues, eigenvectors


def normalize_wfn(psi,x_grid,n):
    for i in range(0,n):
       WF_norm = simps(np.abs(psi[:,i])**2, x=x_grid)
       psi[:,i] /= np.sqrt(WF_norm)
        
    return psi



def side_wp(psi,well_side):
    
    """
    identify which side of the well the wavepacket is.
    """
    side=np.zeros(len(psi[0,:]))
    for i in range(len(psi)):
        ind = np.argmax(psi[:,i])
        if ind < 100 :
            well=-1
        else : 
            well=1
        side[i]=well
        
    #indices of all wells on the left side (with -1 in side)
    indices=[i for i, x in enumerate(side) if x == well_side]
    
    #take the lowest index
    if well_side ==-1 :
        val_tk=indices[0]
    else : 
        val_tk=indices[0:4]
    return psi[:,val_tk]
