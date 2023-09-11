import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import numpy as np

@tf.function
def NS3D(model, coords, params, separate_terms=False):
    """ NS 3D equations """

#    PX    = params[0]
    g    = params[0]

    with tf.GradientTape(persistent=True) as tape1: #no necesito 2nd derivatives
        tape1.watch(coords)
        Yp = model(coords)[0]
        u  = Yp[:,0] #ux
        v  = Yp[:,1] #uy
        h  = Yp[:,2] #h
        #p  = Yp[:,3] #no deberia tambien haber hb?
        hb = Yp[:,3] #sino no hay forma de meter en las ecs

    # First derivatives
    grad_u = tape1.gradient(u, coords)
    u_t = grad_u[:,0]
    u_x = grad_u[:,1]
    u_y = grad_u[:,2]
    u_z = grad_u[:,3]

    grad_v = tape1.gradient(v, coords)
    v_t = grad_v[:,0]
    v_x = grad_v[:,1]
    v_y = grad_v[:,2]
    v_z = grad_v[:,3]

    grad_h = tape1.gradient(h, coords)
    h_t = grad_h[:,0]
    h_x = grad_h[:,1]
    h_y = grad_h[:,2]
    h_z = grad_h[:,3]

    grad_hb = tape1.gradient(hb, coords)
    hb_t = grad_hb[:,0]
    hb_x = grad_hb[:,1]
    hb_y = grad_hb[:,2]
    hb_z = grad_hb[:,3]

    #grad_p = tape1.gradient(p, coords)
    #p_x = grad_p[:,1]
    #p_y = grad_p[:,2]
    #p_z = grad_p[:,3]

    del tape1



    # Equations to be enforced
    if not separate_terms:
        f0 = u_x + v_y + h_z #incompresible, podria agregar irrotacional tambien?
        f1 = (u_t + u*u_x + v*u_y + h*u_z  + g*h_x)
        f2 = (v_t + u*v_x + v*v_y + h*v_z  + g*h_y)
        f3 = (h_t + u*(h_x-hb_x) + u_x*(h-hb) + v*(h_y-hb_y) + v_y*(h-hb)   )

        return [f0, f1, f2, f3h
    else:
        return ([u_x, v_y, w_z],
                [u_t,
                 u*u_x, v*u_y, w*u_z,
                 p_x, PX*tf.ones(p_x.shape, dtype=p_x.dtype),
                -nu*u_xx, -nu*u_yy, -nu*u_zz],
                [v_t,
                 u*v_x, v*v_y, w*v_z,
                 p_y, 0*tf.ones(p_y.shape, dtype=p_y.dtype),
                -nu*v_xx, -nu*v_yy, -nu*v_zz],
                [w_t,
                 u*w_x, v*w_y, w*w_z,
                 p_z, 0*tf.ones(p_z.shape, dtype=p_z.dtype),
                -nu*w_xx, -nu*w_yy, -nu*w_zz],
                )
