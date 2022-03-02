'''
RK4 routine implemented from NUMERICAL RECIPES IN FORTRAN

TODO:Implement adaptive step size
'''
import numpy as np

def rk4(y, dydx, x, h, derivs):
    '''
    Performs a single 4th order Runge-Kutta step on a set of n differential equations.


    Parameters
    ----------
    y : np.ndarray
        array containing initial values (values at x) for the dependant variables 
    
    dydx : np.ndarray
        array containing initial derivative values (values at x) for each ODE 

    x : float
        initial value for the independant variable
    
    h : float
        step size to be used for integration step 
    
    derivs : function
        external function which evaluates the ODE's derivatives at points x,y_i and returns an array dydx containing their values
    

    Returns
    -------
    yout : np.ndarray
        array containing final values for the dependant variables
    
    '''
    # evaluate RK4 coeffs
    k1 = h * dydx
    k2 = h * derivs(x + h/2, y + k1/2)
    k3 = h * derivs(x + h/2, y + k2/2)
    k4 = h * derivs(x + h, y + k3)

    yout = y + k1/6 + k2/3 + k3/3 + k4/6

    return yout




def rkdumb(ystart, x1, x2, nsteps, derivs):
    '''
    starting from initial values ystart known at starting point x1, uses fourth-order Runge-Kutta to advance nstep equal increments to final point x2.
    Results returned as an array with rows containing integrated values of the dependant variables at position x [x, y_0, y_1, y_2, ..., y_n]

    Parameters
    ----------
    ystart : np.ndarray
        array of initial values for dependant variables
    
    x1 : float
        initial value for the independant variable
    
    x2 : float
        final value for the independant variable
    
    nsteps : int
        number of steps to take integrating equations from starting x1 to final x2
    
    derivs : function
        External function which evaluates ODEs at points x,y and returns an array dydx containing their values
    

    Returns
    -------
    ys : np.ndarray
        array containing integrated values of dependant variables from x1 to x2
    
    '''
    # initialise results array and dep var array y
    y = ystart
    initial_values = np.asarray([x1,*ystart])
    ys = np.zeros((nsteps+1, len(initial_values))) # 
    ys[0] = initial_values
    
    # calculate step size
    h = (x2 - x1) / (nsteps - 1)


    for i in range(nsteps):
        # increment independant variable x 
        x = x1 + i*h
        dAlldx = derivs(x,y)
        # perform an RK4 step to obtain y at x + h
        print(f'\r[INFO] performing rk4 step {i+1}/{nsteps}...',end='')
        y = rk4(y, dAlldx, x, h, derivs)
        # add results to ys array
        ys[i+1] = np.asarray([[x+h,*y]])
    
    return ys
 
