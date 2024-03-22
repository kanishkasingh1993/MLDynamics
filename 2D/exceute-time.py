import torch
import numpy as np
from neuraloperator.model import DeepONetCartesianProd, FNO3d
from helper import plot_his
import timeit

num_repeat = 10000

#mctdh
mctdh_runtime = np.loadtxt('time.txt')
mctdh_mean, mctdh_std = np.mean(mctdh_runtime), np.std(mctdh_runtime)
print(mctdh_mean, mctdh_std)

#DON
x_branch_layers = [4225, 128, 128, 128, 128, 128]
x_trunk_layers = [3, 128, 128, 128, 128, 128]
DON = DeepONetCartesianProd(x_branch_layers, x_trunk_layers)
DON = torch.nn.DataParallel(DON)
DON = DON.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
input_DON = (torch.rand(1,4225).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), torch.rand(1,51*65*65,3).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))

stmt = "DON(x)"
setup = "x = input_DON"
DON_runtime = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())
DON_mean, DON_std = np.mean(DON_runtime), np.std(DON_runtime)
print(DON_mean, DON_std)

torch.cuda.empty_cache()

#FNO
FNO = FNO3d(width=40, modes1=20, modes2=20, modes3=20, lifting_input = 5, output_ch = 1)
FNO = FNO.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
input_FNO = torch.rand(1,5,51,65,65).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
stmt = "FNO(x)"
setup = "x = input_FNO"
FNO_runtime = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())
FNO_mean, FNO_std = np.mean(FNO_runtime), np.std(FNO_runtime)
print(FNO_mean, FNO_std)

m = [mctdh_mean, DON_mean, FNO_mean]
s = [mctdh_std, DON_std, FNO_std]
l = ['MCTDH', 'DeepONet', 'FNO']
plot_his(m, s, l, 'Execution time (second)', '', '/home/leek97/ML/picture/time.png',log=True)