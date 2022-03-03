'''
routines rk4, rkdumb, rkck, rkqs, odeint based on implementation described in NUMERICAL RECIPES IN FORTRAN chapter 16 

'''
import numpy as np
import sys
from icecream import  ic

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
        dydx = derivs(x,y)
        # perform an RK4 step to obtain y at x + h
        print(f'\r[INFO] performing rk4 step {i+1}/{nsteps}...',end='')
        y = rk4(y, dydx, x, h, derivs)
        # add results to ys array
        ys[i+1] = np.asarray([[x+h,*y]])
    
    return ys
 
 
def rkqs(y, dydx, x, htry, eps, yscal, derivs):
    '''
    Makes a 5th order Runge-Kutta step while monitoring truncation error and makes changes to step size to improve efficiency while maintaining specified accuracy (set by eps and yscal) in dependant variables over integration step. 
    Returns updated values for dependant vars, y along with step size used and to be used in next step.
    
    Parameters
    ----------

    y : np.ndarray
        array containing initial values (values at x) for the dependant variables 
    
    dydx : np.ndarray
        array containing initial derivative values (values at x) for each ODE 

    x : float
        initial value for the independant variable
    
    htry : float
        initial step size to use in integration step 
    
    eps : float 
        overall tolerance level used to define minimum truncation error allows per step

    yscal : np.ndarray 
        array setting required accuracy for each dependant variable when multiplied with eps in the step
    
    derivs : function
        external function which evaluates the ODE's derivatives at points x,y_i and returns an array dydx containing their values

    Returns
    -------
    yout : np.ndarray
        array containing integrated values of dependant variables from x1 to x2
    
    hdid : float
        step size actually used in integration step 
    
    hnext : float
        estimated step size to be used in next integration step
    '''
    
    # define params for rkqs step 

    SAFETY = 0.9 # factor accounting for inexactness in truncation error estimation per step
    PGROW = -0.2 # exponent used during step size increase 
    PSHRINK = -0.25 # exponent used during step size decrease
    ERRCON = (5/SAFETY)**(1/PGROW)  # 

    # set step size to htry
    h = htry
    
    # take an integration step
    yout, yerr = rkck(y, dydx, x, h, derivs)
    
    # evaluate accuracy of step
    errmax = 0

    errmax = np.amax([errmax,*abs(yerr/yscal)])
    errmax /= eps # scale error relative to required tolerance level

    while errmax > 1:
        # truncation error too large, reduce step size
        htemp = SAFETY*h*(errmax**PSHRINK)
        h = np.amax([abs(htemp), 0.1*abs(h)]) * np.sign(h) # never reduce by more than factor of 10
        xnew = x+h 
        if (xnew == x): 
            prnt('[ERROR] step size underflow, try relaxing tolerances') 
            sys.exit(1)
        # retry step with new step size
        yout, yerr = rkck(y, dydx, x, h, derivs)
        # re estimate error
        errmax = 0
        errmax = np.amax([errmax,*abs(yerr/yscal)])
        errmax /= eps
    
    # step succeeded, attempt to increase step size
    if errmax > ERRCON:
        hnext = SAFETY*h*(errmax**PGROW)
    else:
        hnext = 5*h # no more than factor 5 increase
    hdid = h 
    x = x+h
    return yout, hdid, hnext

def rkck(y, dydx, x, h, derivs):
    '''
    Makes a 5th order Cash-Karp Runge-Kutta of step h, returns estimation of local truncation error, yerr, and updated values for dependant variables, y. 

    Parameters
    ----------

    y : np.ndarray
        array containing initial values (values at x) for the dependant variables 
    
    dydx : np.ndarray
        array containing initial derivative values (values at x) for each ODE 

    x : float
        initial value for the independant variable
    
    htry : float
        initial step size to use in integration step 
    
    eps : float 
        overall tolerance level used to define minimum truncation error allows per step

    yscal : np.ndarray 
        array setting required accuracy for each dependant variable when multiplied with eps in the step
    
    derivs : function
        external function which evaluates the ODE's derivatives at points x,y_i and returns an array dydx containing their values

    Returns
    -------
    yout : np.ndarray
        array containing integrated values of dependant variables from x1 to x2
    
    yerr : np.ndarray
        array containing estimated local truncation error for each dependant variable in integrating over step.
    '''

    # Define Cash Karp params for 5th order Embedded Runga-Kutta method

    param_matrix = np.asarray([
        [0, 0, 0, 0, 0, 0, 37/378, 2825/27648],
        [1/5, 1/5, 0, 0, 0, 0, 0, 0],
        [3/10, 3/40, 9/40, 0, 0, 0, 250/621, 18575/48384],
        [3/5, 3/10, -9/10, 6/5, 0, 0, 125/594,13525/55296],
        [1, -11/54, 5/2, -70/27, 35/27, 0, 0, 277/14336],
        [7/8, 1631/55296, 175/512, 575/13824, 44275/110592, 253/4096, 512/1771, 1/4]
        ])
    
    a = param_matrix[:,0]
    b = param_matrix[:,1:6]
    c = param_matrix[:,6]
    cstar = param_matrix[:,7]
    # calculate k param values
    k1 = h * dydx
    k2 = h * derivs(x + a[1]*h, y + b[1,0] * k1)
    k3 = h * derivs(x + a[2]*h, y + b[2,0] * k1 + b[2,1] * k2)
    k4 = h * derivs(x + a[3]*h, y + b[3,0] * k1 + b[3,1] * k2 + b[3,2] * k3)
    k5 = h * derivs(x + a[4]*h, y + b[4,0] * k1 + b[4,1] * k2 + b[4,2] * k3 + b[4,3] * k4)
    k6 = h * derivs(x + a[5]*h, y + b[5,0] * k1 + b[5,1] * k2 + b[5,2] * k3 + b[5,3] * k4 + b[5,4] * k5)

    K = np.asarray([k1,k2,k3,k4,k5,k6])
    
    # reshape so the broadcasting works out
    c = c.reshape(-1,1) 
    cstar = cstar.reshape(-1,1)
    y = y.reshape(1,-1)

    
    # calculate final y and estimate truncation error
    ck = c*K
    yout = y + ck[0] + ck[1] + ck[2] + ck[3] + ck[4] + ck[5]
    yerr = np.sum((c - cstar)*K,axis=0)

    return yout.squeeze(), yerr.squeeze() # remove the added dim

    


def odeint(ystart, x1, x2, eps, h1, hmin, derivs, rkqs):
    '''
    Integrates a set of ODEs from x1 to x2 using the rkqs function (adaptive step size cash karp RK-5 method). 
    Tracks number of good and bad steps taken.
    returns array containing integated results at each step along with number of good and bad steps, and the array of step sizes taken. 
    
    Parameters
    ----------

    ystart : np.ndarray
        array of initial values for dependant variables
    
    x1 : float
        initial value for the independant variable
    
    x2 : float
        final value for the independant variable
    
    eps : float
        overall tolerance level used to define minimum truncation error allows per step

    h1 : float
        step size for first step

    hmin : float
        minimum allows step size

    derivs : function(x,y)
        External function which evaluates ODEs at points x,y and returns an array dydx containing their values
    
    rkqs : function(y, dydx, x, htry, eps, yscal, derivs):
        Function which makes a 5th order adaptive step size Runge-Kutta step 
    

    Returns
    -------
    ys : np.ndarray
        array containing integrated values of dependant variables from x1 to x2
    
    nok : int
        number of successful steps taken
    
    nbad : int
        number of bad (but retried and fixed) steps taken
    
    hsteps : np.ndarray
        array of step sizes used  
    '''
    # set odeint hyper params
    MAXSTP = int(1e7) # maximum number of steps allowed to be taken
    TINY = np.finfo(float).resolution # a small number to prevent divide by 0 errors when scaling truncation error
    
    # set the initial values and initalise results array
    x = x1
    y = ystart
    h = np.sign(x2 - x1) * h1

    ys = np.zeros((MAXSTP+1,len([x,*ystart])))
    ys[0] = np.asarray([x1,*ystart])
    hsteps = np.zeros((MAXSTP))
    nok = 0
    nbad = 0

    # take steps
    for i in range(MAXSTP):
        print(f'\r[INFO] performing rk4 step {i}...',end='')
        dydx = derivs(x,y) 
        yscal = abs(y) + abs(h*dydx) + TINY # array used to scale error in rkqs
        if (x+h-x2) * (x+h-x1) > 0:
            # if step size can overshoot, decrease step size
            h = x2 - x 
        
        # make step
        y, hdid, hnext = rkqs(y, dydx, x, h, eps, yscal, derivs)
        
        # keep track of step size used in step
        hsteps[i] = hdid
        if hdid == h: nok+=1
        else: nbad+=1
        
        if ((x-x2) * (x2-x1) > 0):
            # finished? remove 0 rows in ys and hsteps
            ys = ys[~np.all(ys == 0, axis=1)]
            hsteps = hsteps[~np.all(hsteps == 0, axis=0)]

            return ys, nok, nbad, hsteps 
        
        if abs(hnext) < hmin: 
            print('[ERROR] step size to be taken smaller than minimum allowed by odeint')
            sys.exit(1)
        
        # set next step and store results from current step
        h = hnext
        x+= hnext
        ys[i] = np.asarray([x,*y])
    
    print('[ERROR] too many steps in odeint, try reducing required accuracy')
    sys.exit(1)





     




    







        




    

