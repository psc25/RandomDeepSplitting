import numpy as np
import tensorflow as tf
import scipy.stats as sst
from DeepSplittingJump import DSPL_model
import os
import time
import shutil

path = os.path.join(os.getcwd(), "default_merton/")
tf.enable_eager_execution()

dd = [10, 50, 100, 500, 1000, 5000, 10000]

N = 12
M = 500
T = 1.0/3.0

x0 = 30.0
mu0 = -0.01
sigma0 = 0.15
lam = 0.2
muZ = -0.05
sigmaZ = 0.1
delta = 0.1
M0 = 200

beta, R = 2.0/3.0, 0.02
vh, vl = 25.0, 50.0
gammah, gammal = 0.2, 0.02
def f(t, x, y):
    return -(1.0-beta)*np.fmin(np.fmax((y-vh)*(gammah-gammal)/(vh-vl)+gammah, gammal), gammah)*y - R*y

def g(x):
    return np.min(x, axis = -1, keepdims = True)

runs = 10
epochs = 2000
activation = tf.nn.tanh
lr_rate = [1e-2, 1e-3, 1e-4]
lr_bd = [500, 1000]

print("======================================================================")
for di in range(len(dd)):
    d = dd[di]
    
    def mu(t, x):
        return (mu0+sigma0**2/2+lam*(tf.exp(muZ+sigmaZ**2/2)-1-muZ))*x
    
    def sigmadiag(t, x):
        return sigma0*x
    
    def eta(t, x, z):
        return np.expand_dims(x, 1)*(np.exp(z)-1.0)
    
    distr = sst.norm(loc = mu0, scale = sigma0)
    nuAdelta = lam*np.mean(np.linalg.norm(distr.rvs(size = [5000, d]), axis = -1) >= delta)
    def Zdelta(size):
        Z = distr.rvs(size = size)
        if np.prod(Z.shape) > 0.0:
            ind = np.linalg.norm(Z, axis = -1) < delta
            while np.sum(ind) > 0.0:
                Z[ind] = distr.rvs(size = [np.sum(ind), d])
                ind = np.linalg.norm(Z, axis = -1) < delta
                
        return Z.astype(np.float32)
    
    hid_layer_size = [min(d, 2000)]
    
    sol = np.zeros([runs, 1])
    tms = np.zeros([runs, 1])
    fev = np.zeros([runs, 1])
    for ri in range(runs):
        b = time.time()
        dspl = DSPL_model(T, N, M, d, x0, mu, sigmadiag, eta, Zdelta, M0, nuAdelta, f, g, hid_layer_size, activation, path)
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