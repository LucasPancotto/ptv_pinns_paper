
import numpy as np



def oscillator(dd, w0, xx):
    """Analytical solution to the 1D underdamped harmonic oscillator problem"""
    wt = np.sqrt(w0**2-dd**2)
    phi = np.arctan(-dd/wt)
    A = 1/(2*np.cos(phi))
    osc = np.cos(phi+wt*xx)
    damp = np.exp(-dd*xx)
    yy  = damp*2*A*osc
    return yy

def xdot(dd, w0, xx):
    """Time derivative of the analytical solution to the 1D underdamped harmonic oscillator problem"""
    wt = np.sqrt(w0**2-dd**2)
    phi = np.arctan(-dd/wt)
    A = 1/(2*np.cos(phi))
    osc = np.cos(phi+wt*xx)
    cso = np.sin(phi+wt*xx)
    damp = np.exp(-dd*xx)
    yy  = damp*2*A*osc
    return -dd*yy - wt*damp*2*A*cso
