import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import numpy as np

@tf.function
def NS3D(model, coords, params, separate_terms=False):
    """ NS 3D equations """

    dd    = params[0]
    w0    = params[1]



    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(coords)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(coords)
            yp = model(coords)[0]


        # First derivatives
        grad_y = tape1.gradient(yp, coords) #coords [t,y]
        dy = grad_y[:,0] #grad[: ,0] = d /dt
        del tape1


    # Second derivatives
    dy2 = tape2.gradient(dy , coords)[:,0]
    del tape2

    # Equations to be enforced
    f0 = dy2 + (2*dd)*dy + (w0**2)*yp
    return [f0]
