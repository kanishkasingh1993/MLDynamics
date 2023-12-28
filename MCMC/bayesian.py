#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
from scipy.integrate import simps
import scipy.stats as stats



# In[ ]:


def likelihood(overlap_predicted):
    """
    Calculates the likelihood of the observed data given the predicted overlap.
    """
    return stats.norm.pdf(1.0, loc=overlap_predicted, scale=0.1)


# In[ ]:


def pop_i (x_grid,psi_ref,psi_sim):
    
    """
    Calculates the overlap of a wavefunction with another.
    """
    
    overlap= abs(x_grid[0]-x_grid[1])*np.sum(np.conj(psi_sim) * psi_ref)
    fitness=np.abs(overlap**2)
    
    return fitness
    


# In[ ]:


# Define the prior sampler function
def prior(param_1_prior,param_2_prior,param_3_prior):
    """
    Samples parameter values from the prior distribution.
    """
    param_1 = param_1_prior.rvs()
    param_2 = param_2_prior.rvs()
    param_3 = param_3_prior.rvs()
    return [param_1, param_2,param_3]


# In[ ]:




