import numpy as np
import tensorflow as tf
import time
import os

class DSPL_model():
    def __init__(self, T, N, M, d, x0, mu, sigmadiag, eta, Zdistr, M0, nuAdelta, f, g, hid_layer_size, activation, path):
        self.N = N
        self.dt = T/N
        self.M = M
        self.d = d
        self.x0 = x0
        self.mu = mu
        self.sigmadiag = sigmadiag
        self.eta = eta
        self.Zdistr = Zdistr
        self.M0 = M0
        self.nuAdelta = nuAdelta
        self.f = f
        self.g = g
        self.hid_layer_size = hid_layer_size
        self.activation = activation
        self.path = path
        self.counts = np.zeros(shape = 1, dtype = np.int64)
        if not os.path.exists(self.path + "models"):
            os.makedirs(self.path + "models")
            
    def sde(self, n):
        y0 = self.x0*np.ones([self.M, self.d], dtype = np.float32)
        y1 = y0
        W = np.random.normal(size = [self.M, n+1, self.d], scale = np.sqrt(self.dt)).astype(dtype = np.float32)
        P = np.random.poisson(size = (self.M, n+1, 1), lam = self.nuAdelta*self.dt).astype(dtype = np.int32)
        for t in range(n+1):
            drift = self.mu(t*self.dt, y0)*self.dt
            diffu = self.sigmadiag(t*self.dt, y0)*W[:, t]
            
            Pmax = np.max(P[:, t])
            if Pmax > 0:
                Z = self.Zdistr.rvs(size = (self.M, Pmax, self.d)).astype(dtype = np.float32)
                mask = np.tile(np.expand_dims(np.arange(1, Pmax+1), 0), (self.M, 1)) <= np.tile(P[:, t], (1, Pmax))
                jump1 = np.sum(self.eta(t*self.dt, y0, Z)*np.expand_dims(mask, -1), axis = 1)
            else:
                jump1 = np.zeros([self.M, self.d])
            
            V = self.Zdistr.rvs(size = (1, self.M0, self.d)).astype(dtype = np.float32)
            jump2 = self.dt*self.nuAdelta*np.mean(self.eta(t*self.dt, y0, V), axis = 1)
            
            y0 = y1
            y1 = y0 + drift + diffu + jump1 - jump2
            self.counts = self.counts + self.M*self.d + 2
            
        return y0, y1
        
    def NN_init(self):
        layers = []
        for i, layer in enumerate(self.hid_layer_size):
            if i == 0:
                layers.append(tf.keras.layers.Dense(layer, input_shape = (self.d, ), activation = self.activation, dtype = tf.float32))
                self.counts = self.counts + (self.d + 1)*layer
            else:
                layers.append(tf.keras.layers.Dense(layer, activation = self.activation, dtype = tf.float32))
                self.counts = self.counts + (self.hid_layer_size[i-1] + 1)*layer
                
        layers.append(tf.keras.layers.Dense(1, activation = 'linear', dtype = tf.float32))
        self.counts = self.counts + self.hid_layer_size[-1] + 1
        return tf.keras.models.Sequential(layers)

    def train(self, epochs, lr_rate, lr_bd, print_details = False):
        losses = np.nan*np.ones([self.N, epochs])
        self.counts = self.counts + self.M
        # n > 0
        for n in range(self.N-1, 0, -1):
            tf.reset_default_graph()
            yn, yn1 = self.sde(n)
            mdl = self.NN_init()
            if n == self.N-1:
                vn1 = self.g(yn1)
            else:
                mdl1 = tf.keras.models.load_model(self.path + "models/model_" + str(n+1), compile = False)
                vn1 = mdl1(yn1)
                
            df = self.dt*self.f((n+1)*self.dt, yn1, vn1)
            self.counts = self.counts + self.M
            global_step = tf.Variable(0, trainable = False)
            lr1 = tf.cond(tf.less(global_step, lr_bd[0]), lambda: lr_rate[0], lambda: lr_rate[1])
            lr2 = tf.cond(tf.less(global_step, lr_bd[1]), lambda: lr1, lambda: lr_rate[2])
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr2)      
            for i in range(epochs):
                begin = time.time()
                with tf.GradientTape() as tape:
                    vn = mdl(yn)
                    loss = tf.reduce_mean(tf.square(vn - (vn1 + df)))
                    
                grad = tape.gradient(loss, mdl.trainable_weights)
                optimizer.apply_gradients(zip(grad, mdl.trainable_weights))
                losses[n, i] = loss.numpy()
                end = time.time()
                if print_details:
                    print("n = {}, Step {}, time {}s, loss {:g}".format(n+1, i+1, round(end-begin, 1), losses[n, i]))
                
            mdl.save(self.path + "models/model_" + str(n))
                    
        # n = 0
        tf.reset_default_graph()
        yn, yn1 = self.sde(0)
        mdl1 = tf.keras.models.load_model(self.path + "models/model_" + str(1), compile = False)
        vn1 = mdl1(yn1)
        df = self.dt*self.f(self.dt, yn1, vn1)
        self.counts = self.counts + self.M
        sol = np.mean(vn1 - df)
        losses[0, 0] = np.mean(np.square(sol - (vn1 + df)))
        if print_details:
            print("n = {}, loss {:g}".format(1, losses[0, 0]))
                    
        return losses, sol, self.counts