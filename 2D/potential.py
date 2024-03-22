import matplotlib.pyplot as plt
import numpy as np
import torch

data = torch.load('/home/leek97/ML/result/hh_FNO_dens-1-true_out.pt').reshape(100,51,65,65)
X, Y = np.meshgrid(np.linspace(-9,9,65), np.linspace(-9,9,65))
def henon_heiles(x,y,l):
    return 0.5*(x**2)+0.5*(y**2)+l*((x**2)*y-(1/3)*y**3)
V_pot = henon_heiles(X,Y, 0.111803)
level = np.linspace(0, np.sqrt(120), 10)
level = level**2
mycmap = plt.get_cmap('hot_r')
mycmap2 = plt.get_cmap('hot')
line = 1.0
style = 'dashed'
fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.15, 0.125, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 
cp = ax.contour(X, Y, V_pot, level, cmap=mycmap,linestyles=style,linewidths=line)
wf = ax.pcolormesh(X,Y,data[69][0],shading='auto', cmap=mycmap2)

fmt = {}
for v in cp.levels:
    fmt[v] = str(round(v,2)) + 'a.u.'

# Label every other level using strings
ax.clabel(cp, cp.levels, inline=True, fmt=fmt, fontsize=12)
ax.set_xlabel('$q_1$ (a.u.)', fontsize=16)
ax.set_ylabel('$q_2$ (a.u.)', fontsize=16)
cbar = fig.colorbar(wf, ax=ax)
cbar.set_label('$|\psi|^2$', fontsize=16)
plt.savefig('/home/leek97/ML/picture/pes-1.png')
plt.show()