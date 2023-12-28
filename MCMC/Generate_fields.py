#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import signal



# In[2]:


def random_perturbation (x_size,y_size,coh_x,coh_y,amp_var):
    # Define the grid dimensions

    rng = np.random.RandomState(seed=1)

    # Generate a 2D white noise signal
    noise = rng.randn(y_size, x_size)

    # Define the coherence lengths in each dimension
    #coh_x = 12.2
    #coh_y = 0.27

    # Define the amplitude variance
    #amp_var = 11.9
    
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
    


# In[3]:




# In[ ]:




