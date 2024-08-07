#!/usr/bin/env python
# coding: utf-8

# In[1]:

from scipy.integrate import simps
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import numpy as np


def animate_dyn(xmin,xmax,x,D,V):
    # Create the figure
    fs = 16
    ts = 12
    xLimits = [xmin, xmax]
    yLimits = [-10, 10]
    dLimits = [0, 0.01]
    fig, ax = plt.subplots()
    ax.set_xlim(xLimits)
    ax.set_ylim(yLimits)
    ax.set_xlabel('$x$')
    ax.set_ylabel(r"$|\psi|^2$")
    ax.grid()
    
    ax2 = ax.twinx()
    ax.set_xlim(xLimits)
    ax2.set_ylim(dLimits)
    ax2.set_xlabel('$x$', fontsize = fs)
    ax2.set_ylabel(r"$V$", fontsize = fs)

    # Plot the initial state of the function
    line, = ax.plot(x, D[0], 'r-', label="Density", lw=2)
    lineS2, = ax2.plot(x,V[0], 'b-', label="pot", lw=2)

    time_text = ax.text(0.1, 0.95, "",ha='left', va='top', transform=ax.transAxes,
                         fontsize=15, color='red')
    plt.legend(loc="upper right") 
    plt.rc('xtick', labelsize=ts)  
    plt.rc('ytick', labelsize=ts)  
    # Define the animation function
    def animate(i):
        f=i
        line.set_ydata(D[f])
        lineS2.set_ydata(V[f])
        time_text.set_text("Step: %f" % (0.024*f))
        return line, lineS2

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=300, interval=100, blit=False)

    # Show the animation
    plt.show()
    
    return ani

def plot_eig(x_grid,V_sampled,E1,psi):
    
    N_plot_min = 0  # quantum number of first eigenfunction to plot
    N_plot = 5  # number of eigenfunctions to plot

    WF_scale_factor = (np.max(V_sampled) - np.min(V_sampled))/N_plot
    plt.plot(x_grid, V_sampled, ls="-", c="k", lw=2, label="$V(x)$")

    style_cycler = cycle(["-", "--"])  # line styles for plotting
    color_cyler = cycle(["blue", "red", "gray", "orange", "darkturquoise", "magenta"])

    for i in range(N_plot_min, N_plot_min+N_plot):
        # physically normalize WF (norm = 1)
        WF_norm = simps(np.abs(psi[:,i])**2, x=x_grid)
        print(WF_norm)
        #psi[:,i] /= np.sqrt(WF_norm)
        # higher energy --> higher offset in plotting
        WF_plot =  WF_scale_factor*np.abs(psi[:,i])**2 + E1[i]  # also try plotting real part of WF!
        plt.plot(x_grid, WF_plot, ls=next(style_cycler), lw=1.5, color=next(color_cyler),
                 label="$\psi_{}(x)$".format(i))  
        print("E[%s] = %s"%(i, E1[i]))

    plt.xlabel("$x$")
    plt.legend(loc="best")
    plt.show()
    
def plot_wavepacket(x_grid,V_sampled,psi):

    plt.plot(x_grid, V_sampled, ls="-", c="k", lw=2, label="$V(x)$")
    WF_plot = np.abs(psi)**2   # also try plotting real part of WF!
    plt.plot(x_grid, WF_plot, c="r", label="$\psi_{}(x)$")  
        
    plt.xlabel("$x$")
    plt.legend(loc="best")
    plt.show()   
    
def plot_field_2d(t_grid,elec_field):
    y=elec_field[:,0]
    plt.plot(t_grid, y, c="r", label="$\psi_{}(x)$") 
    plt.xlabel("$x$")
    plt.legend(loc="best")
    plt.show()   

def plot_perturbation(x_array,y_array,z):

    # Plot the resulting perturbation function in 2D
    X, Y = np.meshgrid(x_array, y_array)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, z, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    
    return plt

def comp_dyn(xmin,xmax,x,D,D_true,V, frames= 300, play=True, dt=0.024189):
    # Create the figure
    fs = 20
    ts = 50
    xLimits = [xmin, xmax]
    yLimits = [0, 10]
    dLimits = [0, 100]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(xLimits)
    ax.set_ylim(yLimits)
    ax.set_xlabel('$\phi_d$ (rad)', fontsize = fs)
    ax.set_ylabel(r"$|\psi|^2$", fontsize = fs)
    ax.grid()
    
    ax2 = ax.twinx()
    ax.set_xlim(xLimits)
    ax2.set_ylim(dLimits)
    ax2.set_xlabel('$\phi_d$ (degree)', fontsize = fs)
    ax2.set_ylabel(r"$V$ (meV)", fontsize = fs)
    plt.rc('xtick', labelsize=ts)  
    plt.rc('ytick', labelsize=ts)  
    # Plot the initial state of the function
    line, = ax.plot(x, D[0], 'r-', label="FNO", lw=2)
    lineS2, = ax2.plot(x,V[0], 'b-', label="potential", lw=2)
    lineS3, = ax.plot(x,D_true[0], 'k-', label="Split-operator", lw=2)

    time_text = ax.text(0.1, 0.95, "",ha='left', va='top', transform=ax.transAxes,
                         fontsize=15, color='red')
    #fig.legend() 
    
    # Define the animation function
    def animate(i):
        f=i
        line.set_ydata(D[f])
        lineS2.set_ydata(V[f])
        lineS3.set_ydata(D_true[f])
        time_text.set_text("t = %f ps" % (dt*f))
        return line, lineS2

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=frames, interval=100, blit=False)
    
    if play==True:
        # Show the animation
        plt.show()
    else:
        plt.close()
    
    return ani



