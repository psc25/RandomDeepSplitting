import numpy as np
import scipy.special as ssp
import os
import time
from RandomDeepSplittingJump import RDSPL_model

path = os.path.join(os.getcwd(), "counterparty_expVG/")

dd = [10, 50, 100, 500, 1000, 5000, 10000]

N = 12
M = 500
T = 1/2.0

x0 = 100.0
mu0 = -0.0001
sigma0 = 0.01
alpha = 0.1
kappa = 0.0001
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
    
    c = (0.2168+0.932*d/2.0)/(0.392+d/2.0)
    gamma = 2.0*np.power(d, c)/(1.0+np.power(d, c))
    lambd = gamma*np.sqrt(np.pi)*np.exp(ssp.gammaln(d/2.0+0.5)-ssp.gammaln(d/2.0))/ssp.gamma(1.0/gamma)
    
    # Approximate K_{d/2}(x) by the function in https://arxiv.org/abs/2303.13400
    def CDF(x):
        exp1 = ssp.exp1(np.power(np.sqrt(2*alpha/kappa)*x/lambd, gamma))
        return 2*alpha*exp1/gamma
    
    nuAdelta = CDF(delta)
    
    # Approximate the inverse of E1(x) by the following function
    # see also https://mathematica.stackexchange.com/questions/251068/asymptotic-inversion-of-expintegralei-function
    eulermasc = 0.5772156649
    def invE1(x):
        y1 = -np.log(x)-np.log(-np.log(x))-(np.log(-np.log(x))-1.0)/np.log(x)
        y2 = np.exp(-(x+eulermasc))+np.exp(-2.0*(x+eulermasc))+1.25*np.exp(-3.0*(x+eulermasc))
        y1 = np.expand_dims((x < 0.2043338275)*y1, -1)
        y2 = np.expand_dims((x >= 0.2043338275)*y2, -1)
        y = np.concatenate([y1, y2], axis = -1)
        return np.nansum(y, axis = -1)
    
    def invCDF(x):
        y = np.power(invE1(nuAdelta*gamma*x/(2.0*alpha)), 1/gamma)
        return np.sqrt(kappa/(2.0*alpha))*lambd*y
    
    # We split up Z ~ \nu^d_\delta(dz) into Z ~ R * V, where:
    # the random radius R ~ invCDF(U) is obtained using the inverse transform sampling with U ~ Unif(0,1) (see https://en.wikipedia.org/wiki/Inverse_transform_sampling)
    # the random direction V ~ Unif(S^{d-1}) is uniformly distributed on the sphere S^{d-1} (see https://dl.acm.org/doi/pdf/10.1145/377939.377946)
    def Zdelta(size):
        Y = np.random.normal(size = size)
        V = Y/np.linalg.norm(Y, axis = -1, keepdims = True)
        U = np.random.uniform(low = 0.0, high = 1.0, size = np.append(size[:-1], 1))
        R = invCDF(U)
        Z = R*V
        return Z.astype(np.float32)
    
    # We approximate I_nu by sampling random variables Z from the truncated Levy measure
    Z = Zdelta(size = [5000, d])
    Inu = nuAdelta*np.mean(np.exp(Z)-1.0)
    
    def mu(t, x):
        return (mu0+sigma0**2/2+Inu)*x
    
    def sigmadiag(t, x):
        return sigma0*x
    
    def eta(t, x, z):
        return np.expand_dims(x, 1)*(np.exp(z)-1.0)
    
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