import numpy as np
import tensorflow as tf
from DeepSplitting import DSPL_model
import os
import time
import shutil

path = os.path.join(os.getcwd(), "default_bs/")
tf.enable_eager_execution()

dd = [10, 50, 100, 500, 1000, 5000, 10000]

N = 12
M = 500
T = 1.0/3.0

x0 = 30.0
mu0 = -0.01
sigma0 = 0.15

alph, R = 2.0/3.0, 0.02
vh, vl = 25.0, 50.0
gammah, gammal = 0.2, 0.02
def f(t, x, y):
    return -(1.0-alph)*np.fmin(np.fmax((y-vh)*(gammah-gammal)/(vh-vl)+gammah, gammal), gammah)*y - R*y

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
        return (mu0+sigma0**2/2)*x
        
    def sigmadiag(t, x):
        return sigma0*x
        
    hid_layer_size = [min(d, 2000)]
    
    sol = np.zeros([runs, 1])
    tms = np.zeros([runs, 1])
    fev = np.zeros([runs, 1])
    for ri in range(10):        
        b = time.time()
        dspl = DSPL_model(T, N, M, d, x0, mu, sigmadiag, f, g, hid_layer_size, activation, path)
        _, sol[ri, 0], fev[ri, 0] = dspl.train(epochs, lr_rate, lr_bd, print_details=True)
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