import torch
import numpy as np
from helper import plot_his, error_mean_std, henon_heiles, harmonic
from neuraloperator.loss import H1Loss

X, Y = np.meshgrid(np.linspace(-9,9,65), np.linspace(-9,9,65))
ha = harmonic(X,Y)
hh = henon_heiles(X,Y, 0.111803)
picture_path = '/home/leek97/ML/picture/'
nt, nx, ny = 51, 65, 65

def main(task, model, data_path, n, renorm=False):
    print(renorm)
    m = []
    s = []
    l = []
    for i in range(n):
        l.append(str(i*100+100))
        path = data_path+'-'+str(i)
        pic_path = picture_path+task+'-'+model+'-ntrain'+str(i*100+100)
        if task == 'an':
            ntest = 4080
        else:
            ntest = 100
        mean, std = error_mean_std(task, path, pic_path, ntest, details=True, loss=H1Loss(d=3), renorm = renorm)
        m.append(mean)
        s.append(std)

    plot_his(m, s, l, 'H1Loss ($a.u.^{-2}$)', 'number of training data', picture_path+task+'-'+model+'-trainvs.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('data_path', type=str)
    parser.add_argument('n', type=int)
    parser.add_argument('renorm', type=int)
    args = parser.parse_args()
    main(args.task, args.model, args.data_path, args.n, args.renorm)