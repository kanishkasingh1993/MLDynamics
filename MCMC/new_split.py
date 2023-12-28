get_ipython().run_line_magic('matplotlib', 'notebook')
from IPython.display import HTML, Image
import torch 
import numpy as np
from math import pi,sqrt
import torch
import os
import matplotlib.pyplot as plt
from matplotlib import animation
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from tqdm import tqdm


# In[ ]:





# In[ ]:


class split_op() :
    def __init__(self,xmin,xmax,Nx,nt,dt,t_end,sig=0.1,amp=10.0,
            device=None,dtype=torch.float64):
        #self,xmin=0.0,xmax=1.0,Nx=100,nt=100,dt=0.01,t_end=1.0,sig=0.1,amp=10.0,
        #    device=None,dtype=torch.float64
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
    #    self.wfc0 = np.empty(Nx, dtype=torch.cfloat)
        
        self.u = torch.zeros_like(x, device=device,dtype=torch.cfloat)
        self.u0 = torch.zeros_like(self.u, device=device,dtype=torch.cfloat)
        self.sig=sig
        self.t = 0
        self.it = 0
        self.U = []
        self.T = []
        self.D=[]
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
            
          
    
          #self.V=V_set
          #self.u0=u0
          #self.K=K
          #self.R=R
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


