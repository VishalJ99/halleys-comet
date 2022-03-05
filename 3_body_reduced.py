'''
Script to calculate orbits for 2 planets orbitting central stationary mass M
'''

from unicodedata import ucd_3_2_0
from rk_methods import *
import numpy as np
import matplotlib.pyplot as plt
import argparse

def derivs(t, Y):
    '''
    Calculates values for derivatives for ODEs governing a reduced 3 body problem,
    consisting of 2 objects orbiting a fixed central mass.
    ODEs obtained via newtons law of gravity and newton's laws. 
    
    Parameters
    ----------
    t : float
        value for time at which derivatives are being evaluated (not needed for these equations) 
    
    Y : np.ndarray
        array containing dependant variables: x1,y1,x2,y2,u1,v1,u2,v2 (positions and velocities)  
    
    Returns
    -------
    dAlldx : np.ndarray
        array containing values for derivatives of dependant variables at time t
    
    '''
    # set model params
    G = 6.674e-11
    M = 1.989e30
    m1 = 1e-3*M
    m2 = 4e-2*M

    # unpack dependant variables array: positions x_i,y_i and velocities u_i,v_i
    x1,y1,x2,y2,u1,v1,u2,v2 = Y
    
    # calculate radial seperations
    r1 = (x1**2 + y1**2)**0.5
    r2 = (x2**2 + y2**2)**0.5
    
    x12 = x2 - x1
    y12 = y2 - y1
    r12 = (x12**2 + y12**2)**0.5

    # evaluate derivatives
    dx1dt = u1
    dy1dt = v1

    dx2dt = u2 
    dy2dt = v2 
    
    du1dt = - (G*M / r1**3) * x1 + (G*m2/r12**3) * x12 
    dv1dt = - (G*M / r1**3) * y1 + (G*m2/r12**3) * y12 

    du2dt = - (G*M / r2**3) * x2 - (G*m2/r12**3) * x12 
    dv2dt = - (G*M / r2**3) * y2 - (G*m2/r12**3) * y12

    dAlldt = np.asarray([dx1dt, dy1dt, dx2dt, dy2dt, du1dt, dv1dt, du2dt, dv2dt])
    return dAlldt


def main(t_end, h_0, h_min, eps):
    # q2a
    # initial conditions: time in s, x,y positions in m, u,v velocities in ms^-1
    au = 1.496e11
    x1 = 2.52*au
    y1 = 0.
    u1 = 0. 
    v1 = -1.88e4 
    
    x2 = 5.24*au
    y2 = 0.
    u2 = 0. 
    v2 = -1.304e4  
    t_start = 0

    # integrate equations
    print('-'*30)
    print('Integrating newtons laws of motion for reduced 3 body system')
    print(f'Integrating from {t_start} to {t_end:.3e}')
    print('-'*30)
    
    Y_start = np.asarray([x1,y1,x2,y2,u1,v1,u2,v2])
    
    ys, nok, nbad, hsteps = odeint(Y_start, t_start, t_end, eps, h_0, h_min, derivs, rkqs)
    
    # plot trajectories with alpha based on time (for matplotlib version > 3.4 just set alpha = alphas)
    alpha_linear = np.linspace(0.1,0.5,len(ys))
    alpha_sigmoid = 1/(1 + np.exp(-(15*alpha_linear-10)))
    rgba_colors = np.zeros((len(ys),4))
    # plot planet 1s trajectory with colour red
    rgba_colors[:,0] = 1.0
    rgba_colors[:,3] = alpha_linear
    plt.scatter(ys[:,1],ys[:,2],color = rgba_colors,s=1)
    # plot planet 2s trajectory with colour green
    rgba_colors[:,0] = 0
    rgba_colors[:,1] = 1.0
    plt.scatter(ys[:,3],ys[:,4],color = rgba_colors,s=1)
    plt.title("Y vs X plot for 3 body system")
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Script to calculate orbits for 2 planets orbitting central stationary mass M")
    # lambda functions for type allow for scientific numbers to be passed via command line
    parser.add_argument('--t_end', type=lambda x: float(x), help = 'value for final time in seconds to integrate till from t=0, default value = 2.398e+9 (1 period)', default = 2.398e+9)
    parser.add_argument('--h_0', type=lambda x: int(float(x)), help = 'value for the initial step size (default = 1e5)', default = 1e5)
    parser.add_argument('--h_min', type=lambda x: int(float(x)), help = 'value for the smallest allowed step size (default = 0)', default = 0)
    parser.add_argument('--eps', type=lambda x: float(x), help = 'value for tolerance to set accuracy of each step for adaptive step size algorithm rkqs, (default = 1e-9)', default = 1e-9)

    args = parser.parse_args()
    main(args.t_end, args.h_0, args.h_min, args.eps)
