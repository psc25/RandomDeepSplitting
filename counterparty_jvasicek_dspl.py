import numpy as np
import tensorflow as tf
import scipy.stats as sst
from DeepSplittingJump import DSPL_model
import os
import time
import shutil

path = os.path.join(os.getcwd(), "counterparty_jvasicek/")
tf.enable_eager_execution()

dd = [10, 50, 100, 500, 1000, 5000, 10000]

N = 12
M = 500
T = 1.0/2.0

x0 = 100.0
alpha = 0.01
mu0 = 100.0
sigma0 = 2
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
epochs = 2000
activation = tf.nn.tanh
lr_rate = [1e-2, 1e-3, 1e-4]
lr_bd = [500, 1000]

print("======================================================================")
for di in range(len(dd)):
    d = dd[di]
    
    Zdistr = sst.uniform(loc = 0.0, scale = 1.0)
    nuAdelta = lam*np.mean(np.linalg.norm(Zdistr.rvs(size = [5000, d]), axis = -1) >= delta, dtype = np.float16)
    
    def mu(t, x):
        return alpha*(mu0-x)
    
    def sigmadiag(t, x):
        return sigma0
    
    def eta(t, x, z):
        return z
    
    hid_layer_size = [min(d, 2000)]
    
    sol = np.zeros([runs, 1])
    tms = np.zeros([runs, 1])
    fev = np.zeros([runs, 1])
    for ri in range(runs):
        b = time.time()
        dspl = DSPL_model(T, N, M, d, x0, mu, sigmadiag, eta, Zdistr, M0, nuAdelta, f, g, hid_layer_size, activation, path)
        _, sol[ri, 0], fev[ri, 0] = dspl.train(epochs, lr_rate, lr_bd)
        e = time.time()
        tms[ri, 0] = e-b
        print("Deep Splitting performed for d = " + str(d) + ", run = " + str(ri+1) + "/" + str(runs) + ", in " + str(np.round(tms[ri, 0], 1)) + "s, with solution " + str(sol[ri, 0]))
        
        np.savetxt(path + "det_sol_" + str(d) + ".csv", sol)
        np.savetxt(path + "det_tms_" + str(d) + ".csv", tms)
        np.savetxt(path + "det_fev_" + str(d) + ".csv", fev)
        
        del dspl
        tf.reset_default_graph()

shutil.rmtree(path + "models")
print("======================================================================")
print("Deep Splitting solutions saved")