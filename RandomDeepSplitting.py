import numpy as np
import scipy.linalg as scl
import time

class RDSPL_model():
    def __init__(self, T, N, M, d, x0, mu, sigmadiag, f, g, hid_layer_size, activation, nu):
        self.T = T
        self.N = N
        self.dt = T/N
        self.M = M
        self.d = d
        self.x0 = x0
        self.mu = mu
        self.sigmadiag = sigmadiag
        self.f = f
        self.g = g
        self.hid_layer_size = hid_layer_size
        self.activation = activation
        self.counts = np.zeros(shape = 1, dtype = np.int64)
        self.rnd_wght = np.random.normal(size = [d, hid_layer_size])
        self.rnd_bias = np.random.normal(size = [1, hid_layer_size])
        self.counts = self.counts + hid_layer_size*(d + 1)
        self.y = x0*np.ones([M, N+1, d], dtype = np.float32)
        W = np.random.normal(size = [M, N, d], scale = np.sqrt(self.dt)).astype(np.float32)
        for t in range(N):
            drift = mu(t*self.dt, self.y[:, t])*self.dt
            diffu = sigmadiag(t*self.dt, self.y[:, t])*W[:, t]
            self.y[:, t+1] = self.y[:, t] + drift + diffu
            
        self.counts = self.counts + (M*d + 2)*N

    def RN_init(self):
        rnd_lout = np.zeros([self.hid_layer_size, 1], dtype = np.float32)
        return [self.rnd_wght, self.rnd_bias, rnd_lout, 0.0, 1.0]
    
    def RN_batch_normalization(self, mdl, x):
        hid = np.matmul(x, mdl[0]) + mdl[1]
        mdl[3] = np.mean(hid)
        mdl[4] = np.std(hid)
        return mdl        
        
    def RN_neur(self, mdl, x):
        hid = np.matmul(x, mdl[0]) + mdl[1]
        hid = (hid-mdl[3])/mdl[4]
        return self.activation(hid)
    
    def RN_eval(self, mdl, x):
        return np.matmul(self.RN_neur(mdl, x), mdl[2])
    
    def train(self, print_details = False):
        losses = np.zeros([self.N, 1], dtype = np.float32)
        vn1 = self.g(self.y[:, self.N])
        self.counts = self.counts + self.M
        # n > 0
        for n in range(self.N-1, 0, -1):
            b = time.time()
            mdl = self.RN_init()
            mdl = self.RN_batch_normalization(mdl, self.y[:, n])         
            R_train = self.RN_neur(mdl, self.y[:, n])            
            y_train = vn1 + self.dt*self.f((n+1)*self.dt, self.y[:, n+1], vn1)
            self.counts = self.counts + self.M
            
            mdl[2] = scl.lstsq(R_train, y_train, lapack_driver = 'gelsy')[0]
            vn = self.RN_eval(mdl, self.y[:, n])
            losses[n] = np.mean(np.square(vn - y_train))
            self.counts = self.counts + 2
            e = time.time()
            if print_details:
                print('Step {}, time {}s, loss = {}'.format(n+1, round(e-b, 1), losses[n, 0]))
                
            vn1 = vn
            
        # n = 0
        df = self.dt*self.f(self.dt, self.y[:, 1], vn1)
        self.counts = self.counts + self.M
        sol = np.mean(vn1 + df)
        losses[n] = np.mean(np.square(sol - (vn1 + df)))
        self.counts = self.counts + self.M
        return losses, sol, self.counts