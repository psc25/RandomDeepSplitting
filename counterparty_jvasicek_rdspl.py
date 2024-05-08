import numpy as np
import scipy.stats as sst
import os
import time
from RandomDeepSplittingJump import RDSPL_model

path = os.path.join(os.getcwd(), "counterparty_jvasicek/")

dd = [10, 50, 100, 500, 1000, 5000, 10000]

N = 12
M = 500
T = 1.0/2.0

x0 = 100.0
alpha = 0.01
mu0 = 100.0
sigma0 = 2.0
lam = 0.5
delta = 0.1
M0 = 200

beta = 0.03
def f(t, x, y):
    return -beta*np.fmin(y, 0)

K1 = 80.0
K2 = 100.0
L = 5.0
def g(x):
    return np.fmax(np.min(x, axis = -1, keepdims = True) - K1, 0) - np.fmax(np.min(x, axis = -1, keepdims = True) - K2, 0) - L

runs = 10
activation = np.tanh

print("======================================================================")
for di in range(len(dd)):
    d = dd[di]
    
    def mu(t, x):
        return alpha*(mu0-x)
    
    def sigmadiag(t, x):
        return sigma0
    
    def eta(t, x, z):
        return z
    
    distr = sst.uniform(loc = 0.0, scale = 1.0)
    nuAdelta = lam*np.mean(np.linalg.norm(distr.rvs(size = [5000, d]), axis = -1) >= delta)
    def Zdelta(size):
        Z = distr.rvs(size = size)
        if np.prod(Z.shape) > 0.0:
            ind = np.linalg.norm(Z, axis = -1) < delta
            while np.sum(ind) > 0.0:
                Z[ind] = distr.rvs(size = [np.sum(ind), d])
                ind = np.linalg.norm(Z, axis = -1) < delta
                
        return Z.astype(np.float32)
    
    hid_layer_size = min(d, 2000)
    
    sol = np.zeros([runs, 1])
    tms = np.zeros([runs, 1])
    fev = np.zeros([runs, 1])
    for ri in range(runs):         
        b = time.time()
        rdspl = RDSPL_model(T, N, M, d, x0, mu, sigmadiag, eta, Zdelta, M0, nuAdelta, f, g, hid_layer_size, activation)
        _, sol[ri, 0], fev[ri, 0] = rdspl.train()
        e = time.time()
        tms[ri, 0] = e-b
        print("Random Deep Splitting performed for d = " + str(d) + ", run = " + str(ri+1) + "/" + str(runs) + ", in " + str(np.round(tms[ri, 0], 1)) + "s, with solution " + str(sol[ri, 0]))
        
    np.savetxt(path + "rnd_sol_" + str(d) + ".csv", sol)
    np.savetxt(path + "rnd_tms_" + str(d) + ".csv", tms)
    np.savetxt(path + "rnd_fev_" + str(d) + ".csv", fev)

print("======================================================================")
print("Random Deep Splitting solutions saved")