# TF2.0 LES Channel test

import tensorflow as tf
from   tensorflow import keras
tf.keras.backend.set_floatx('float32')

from pinn      import PhysicsInformedNN

from osc import oscillator, xdot

from   dom import *
from   mod import *
import numpy as np
import time
import random

# Get parameters
params = Run()
#params.add_params(f"data/dset.py")

# NN params
layers = [2]+[params.hu]*params.layers+[2]

# Load data
# Generate data

dd = 2
w0 = 20
xx = np.linspace(0, 1, num=500)
yy = oscillator(dd, w0, xx)

# Training data
X_data = xx[0:200:20].reshape(-1, 1)
Y_data = yy[0:200:20].reshape(-1, 1)
#X_data, Y_data = generate_data(params)

# Normalization layer
inorm = [X_data.min(0), X_data.max(0)]
means     = Y_data.mean(0)
means[-1] = params.P
stds      = Y_data.std(0)
stds[-1]  = params.sig_p
onorm = [means, stds]

# Optimizer scheduler
dsteps = params.depochs*len(X_data)/params.mbsize
params.lr = keras.optimizers.schedules.ExponentialDecay(params.lr,
                                                        dsteps,
                                                        params.drate)

# Initialize model
from equations import NS3D as Eqs
eq_params = ([dd, w0])
eq_params = [np.float32(p) for p in eq_params]

PINN = PhysicsInformedNN(layers,
                         dest=params.paths.dest,
                         norm_in=inorm,
                         norm_out=onorm,
                         optimizer=keras.optimizers.Adam(params.lr),
                         eq_params=eq_params)

# Validation method
PINN.validation = dns_validation(PINN, params, Eqs, eq_params)

# Train
PINN.train(X_data, Y_data,
           Eqs,
           epochs=5,
           batch_size=params.mbsize,
           alpha=0.1,
           print_freq=10,
           valid_freq=100,
           save_freq=10)
