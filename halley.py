'''
Integrates newtons law of motion for comet Halley

TODO: Figure out ways to visualise plot with time based on color / alpha vals
'''

from rk_methods import rk4,rkdumb 
import numpy as np
import matplotlib.pyplot as plt
import argparse

def derivs(t, Y):
    '''
    defines ODEs to integrate to evaluate trajectory of Halley's comet
    ODEs obtained via newtons law of gravity and newtons 2nd law on Halley's comet.
    Assumes central mass M, about which comet is orbiting, is stationary. 
    
    Parameters
    ----------
    t : float
        value for time at which derivatives are being evaluated 
    
    Y : np.ndarray
        array containing dependant variables: x,y,U,V (positions and velocities)  
    
    Returns
    -------
    dydx : np.ndarray
        array containing values for derivatives of ODEs
    
    '''
    G = 6.674e-11
    M = 1.989e30
    
    # unpack dependant variables array: positions x,y and velocities U,V
    x,y,U,V = Y
    
    # calculate radial seperation
    r = (x**2 + y**2)**0.5

    # evaluate derivatives
    dxdt = U
    dydt = V

    dUdt = - (G*M / r**3) * x
    dVdt = - (G*M / r**3) * y

    dYdt = np.asarray([dxdt, dydt, dUdt, dVdt])
    return dYdt

def main(t_end, n_steps):
    # initial conditions: time in s, positions in m, velocities in ms^-1
    x = 5.2e12  
    y = 0.

    U = 0. # x component of velocity
    V = -880 # y component of velocity
    
    t_start = 0

    # integrate equations
    print('-'*30)
    print('Integrating newtons laws of motion for Halleys comet')
    print(f'Initial position:({x:.3e},{y:.3e}), initial velocities ({U:.3e},{V:.3e})')
    print(f'Integrating from {t_start} to {t_end:.3e} using {n_steps} steps')
    print('-'*30)
    
    Y_start = np.asarray([x,y,U,V])
    ys = rkdumb(Y_start, t_start, t_end, n_steps, derivs)

    # plot trajectory with alpha based on time (for matplotlib version > 3.4 just set alpha = alphas)
    alpha_linear = np.linspace(0.1,0.5,len(ys))
    alpha_sigmoid = 1/(1 + np.exp(-(15*x-10)))
    rgba_colors = np.zeros((len(ys),4))
    rgba_colors[:,0] = 1.0
    rgba_colors[:, 3] = alpha_linear
    
    plt.title("Y vs X plot for Halley's comet")
    plt.scatter(ys[:,1],ys[:,2],color = rgba_colors)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Script to calculate trajectory of Halley's comet")
    parser.add_argument('--t_end', type = float, help = 'value for final time in seconds to integrate till from t=0, default value = 2.398e+9 (1 period)', default = 2.398e+9)
    parser.add_argument('--n_steps', type = int, help = 'value for the number of steps to take to integrate the equations (default = 5000)', default = 5000)

    args = parser.parse_args()
    main(args.t_end, args.n_steps)

    
