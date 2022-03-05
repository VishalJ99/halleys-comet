import numpy as np
from rk_methods import *
import colorsys
from ruamel.yaml import YAML
import argparse
import matplotlib.pyplot as plt
from icecream import ic
'''
TODO: Vectorise calculation of acceleration on each body 

DEBUG - step size under flow error 
What are the derivatives being calculated at the first step?
Is this correct?
Experiment with range of initial steps
'''

class N_body():
    '''
    Class to simulate N body gravitationally bound systems using embedded cash karp rk5 with adaptive step size   
    ...
    
    Attributes
    ----------
   
    G : float
        gravitional constant (m^3 kg^-1 s^-2)
    
    m_array : np.ndarray
        array of masses for each body
    
    N : int
        Number of bodies in system

    Y_start : np.ndarray
        array of values defining initial positions (m) and velocites (ms^-1) of bodies
    
    t_start : float
        initial value for time (s) from which to start integrating
    
    t_final : float
        final value for time (s) at which to stop integrating
    
    h_0 : float
        initial guess for step size to use
    
    eps : float
        tolerance level determining accuracy of each step
    
    Y : np.ndarray
        array of values defining position and velocities of bodies at a particular step
    
    t : float
        value for independant variable - time (s)
    
    dAlldt : np.ndarray
        array containing values for derivatives of positions and velocities at a particular step
    
    results : np.ndarray
        array storing integrated results from tstart to tend

    show_fig : bool
        set to 1 or 0 / True or False if plot method to print fig to screen
    
    save_fig_path : str
        set to path to save figure to for plot method, default = 'N_body_trajectories.png'
        if boolean of argument passed == False will not save, i.e '', 0 or False

    Methods
    -------
    derivs(t,Y)
        calculates values for derivatives of positions and velocities of bodies at time t

    forward():
        Integrates the ODEs using odeint stores results in results array   
    
    trajectory_plot(show_fig, save_fig_path=''N_body_trajectories.png')
        plots trajectories of bodies from results array, each trajectory shown in different colour using opacity as a proxy for time
    
    CoM_trajectory_plot(self, show_fig = 1, save_fig_path='N_body_com_trajectories.png')
        plots trajectories of bodies from CoM ref frame, each trajectory shown in a different colour using opacity as a proxy for time 
    _get_colors(num_colors):
        generate num_colours distinct RGB tuples
    '''
    def __init__(self, m_array, Y_start, t_start, t_final, h_0, h_min, eps):
        '''
        Initialises model
        
        Parameters
        ----------
        Y_start : np.ndarray
        array of values defining initial positions and velocites of bodies
    
        t_start : float
            initial value for time (s) from which to start integrating
        
        t_final : float
            final value for time(s) at which to stop integrating
        
        h_0 : float
            initial guess for step size to use
        
        h_min : float
            smallest allowed step size to use for a step
        
        eps : float
            tolerance level determining accuracy of each step
        '''
        # initialise model parameters
        self.m_array = m_array
        self.N = len(m_array)
        assert (len(Y_start) % 4 == 0) and (len(Y_start) / 4 == self.N), '[ERROR] mismatch between number of bodies specified by mass array and number of dependant variables'
        self.Y_start = Y_start
        self.t_start = t_start
        self.t_final = t_final
        self.h_0 = h_0
        self.h_min = h_min
        self.eps = eps
        self.results = None

        # store indices of positions and velocities for derivs and plot methods
        self.x_indices = np.arange(0,len(Y_start)-3,4)
        self.y_indices = np.arange(1,len(Y_start)-2,4)
        self.u_indices = np.arange(2,len(Y_start)-1,4)
        self.v_indices = np.arange(3,len(Y_start),4)
        
    def derivs(self,t,Y):
        '''
        Calculates derivatives of positions and velocities for N bodies, returns results in an array
        Test effect of vectorisation for solar system 
        
        Parameters
        ----------
        t : float
            value for time at current step (not used in calculating derivatives but required for odeint function)
        Y : np.ndarray
            array containing positions and velocities at current time step [x1,y1,u1,v1,...,xN,yN,uN,vN]        

        Returns
        -------
        dAlldt : np.ndarray
            array containing values for derivatives of positions and velocities at time t
        '''
        
        G = 6.674e-11

        # initalise empty dAlldt array
        dAlldt = np.zeros(Y.shape)
        
        # set velocities as values for derivatives of positions in dAlldt
        dAlldt[self.x_indices] = Y[self.u_indices]
        dAlldt[self.y_indices] = Y[self.v_indices]
        
        # calculate acceleration of each body
        x_pos_array = Y[self.x_indices]
        y_pos_array = Y[self.y_indices]
        # initialise empty arrays to store accelerations on each body
        a_x,a_y = np.zeros(self.N),np.zeros(self.N)
        for i in range(self.N):
            ax_body,ay_body = np.zeros(self.N), np.zeros(self.N)
            # calculate acceleration of ith body
            x,y = x_pos_array[i], y_pos_array[i]
            rel_x_pos_array = x - x_pos_array
            rel_y_pos_array = y - y_pos_array
            for j in range(self.N):
                if i!=j:
                    xij, yij = rel_x_pos_array[j],rel_y_pos_array[j]
                    r = (xij**2 + yij**2)**0.5 
                    ax_body[j] = - (G*self.m_array[j] / r**3 ) * xij
                    ay_body[j] = - (G*self.m_array[j] / r**3) * yij
            # sum over individual accelerations to get net acceleration
            a_x[i] = np.sum(ax_body)
            a_y[i] = np.sum(ay_body)

        # set accelerations as values for derivatives of velocities in dAlldt
        dAlldt[self.u_indices] = a_x
        dAlldt[self.v_indices] = a_y
        # print('Y:',Y)
        # print('dAlldt:',dAlldt)
        # exit(1)
        return dAlldt


    def forward(self):
        '''
        Integrates equations of motion for bodies using odeint 
        returns results array containing values for position and velocity at every time step integrated across

        Parameters
        ----------
        None

        Returns
        -------
        results : np.ndarray
            array storing integrated results from tstart to tend
        
        '''
        ic(self.Y_start, self.m_array)
        results, _, _, _ = odeint(self.Y_start, self.t_start, self.t_final, self.eps, self.h_0, self.h_min, self.derivs, rkqs)
        self.results = results
        return results
    
    def trajectory_plot(self, show_fig = 1, save_fig_path='N_body_trajectories.png'):
        '''
        Plots orbits of every body in CoM frame
        '''
        if self.results is None: print('[WARNING] please call .forward method to generate results before plotting')
        # plot trajectories of each planet on same scatter plot with variable colour alphas as proxy for time
        Y_results = self.results[:,1:]
        x_pos_array = Y_results[:, self.x_indices]
        ic(x_pos_array.shape,x_pos_array[:5])
        y_pos_array = Y_results[:, self.y_indices]
        # generate colours for each bodys trajectory
        rgb_color_list = self._get_colors(self.N)
        rgba_colors = np.zeros((len(self.results),4))
        # use opacity as a proxy for time
        alpha_linear = np.linspace(0.1,0.5,len(self.results))
        alpha_sigmoid = 1/(1 + np.exp(-(15*alpha_linear-10)))
        rgba_colors[:,3] = alpha_linear

        for i in range(self.N):
            # generate colour for each orbit 
            rgba_colors[:,:3] = rgb_color_list[i]
            plt.scatter(x_pos_array[:,i],y_pos_array[:,i],color = rgba_colors,s=1)
        if show_fig: plt.show()
        if save_fig_path: plt.savefig(f"{save_fig_path}")
        plt.cla()
        
        for i in range(self.N):
            # generate colour for each orbit 
            rgba_colors[:,:3] = rgb_color_list[i]
            plt.scatter(x_pos_array[:,i],y_pos_array[:,i],color = rgba_colors,s=1)
        plt.title('velocity plot of bodies')
        if show_fig: plt.show()
        
    def CoM_trajectory_plot(self, show_fig = 1, save_fig_path='N_body_com_trajectories.png'):
        '''
        Plots orbits of every body in CoM frame
        '''
        if self.results is None: print('[WARNING] please call .forward method to generate results before plotting')
        # plot trajectories of each planet on same scatter plot with variable colour alphas as proxy for time
        x_pos_array = self.results[:, self.x_indices]
        y_pos_array = self.results[:, self.y_indices]

        com_x_pos_array = 0
        com_y_pos_array = 0

        # generate colours for each bodys trajectory
        rgb_color_list = self._get_colors(self.N)
        rgba_colors = np.zeros((len(self.results),4))
        # use opacity as a proxy for time
        alpha_linear = np.linspace(0.1,0.5,len(self.results))
        alpha_sigmoid = 1/(1 + np.exp(-(15*alpha_linear-10)))
        rgba_colors[:,3] = alpha_linear

        for i in range(self.N):
            # generate colour for each orbit 
            rgba_colors[:,:3] = rgb_color_list[i]
            plt.scatter(x_pos_array[:,i],y_pos_array[:,i],color = rgba_colors,s=1)
        if show_fig: plt.show()
        if save_fig_path: plt.savefig(f"{save_fig_path}")

        plt.cla()
    
    def _get_colors(self,num_colors):
        '''
        returns list of N 'distinct' colours
        https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors

        Parameters
        ---------
        num_colours : int
            number of colours to generate
        
        Returns 
        -------
        List of containing rgb tuples for num_colours distinct colours
        '''
        colors=[]
        for i in np.arange(0., 360., 360. / num_colors):
            hue = i/360.
            lightness = (50 + np.random.rand() * 10)/100.
            saturation = (90 + np.random.rand() * 10)/100.
            colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
        return colors

def parse_yaml_dict(yaml_dict):
    '''parses a N_body_config.yaml file expects format for each body as follows:
    body1:
        mass: 1.989e30
        x0: 0
        y0: 0
        u0: 0
        v0: -15
    
    Parameters
    ----------
    yaml_dict : dict
        a dictionary containing mass, position and velocity key value pairs for each body
    
    Returns
    -------
    mass_array : np.ndarray
        array containing masses of each body

    Y_start : np.ndarray
        array containing initial positions and velocities of each body
        [x0,y0,u0,v0,...,xN,yN,uN,vN]
    '''
    Y_start = np.zeros(4*len(yaml_dict))
    mass_array = np.zeros(len(yaml_dict))

    for idx,key in enumerate(list(yaml_dict.keys())):
        body = yaml_dict[key]
        mass_array[idx] = body['mass']
        Y0 = [body['x0'],body['y0'],body['u0'],body['v0']]
        Y_start[4*idx:4*idx+4] = Y0

    return mass_array, Y_start



def main(config_path,t_start, t_final, h_0, h_min, eps):
    # parse yaml config for initial values and masses of each body
    yaml=YAML(typ='safe') 
    stream = open(config_path, 'r')
    N_body_dict =  yaml.load(stream) 
    mass_array, Y_start = parse_yaml_dict(N_body_dict)
    # integrate equations
    print('-'*30)
    print(f'Integrating newtons laws of motion for {len(mass_array)} body system')
    print(f'Integrating from {t_start} to {t_final:.3e}')
    print('-'*30)
    
    model = N_body(mass_array, Y_start, t_start, t_final, h_0, h_min, eps)
    model.forward()
    model.trajectory_plot()
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Script to calculate orbits for 2 planets orbitting central stationary mass M")
    # lambda functions for type allow for scientific numbers to be passed via command line
    parser.add_argument('config_path', type = str, help = 'path to yaml config file containing masses, initial positions and velocities for N bodies')
    parser.add_argument('--t_start', type=lambda x: float(x), help = 'value of initial time in seconds to integrate from, default value = 0s', default = 0)
    parser.add_argument('--t_final', type=lambda x: float(x), help = 'value for final time in seconds to integrate till from t=0, default value = 3.156e+9 ', default = 3.156e+9)
    parser.add_argument('--h_0', type=lambda x: int(float(x)), help = 'value for the initial step size (default = 1e5)', default = 1e5)
    parser.add_argument('--h_min', type=lambda x: int(float(x)), help = 'value for the smallest allowed step size (default = 0)', default = 0)
    parser.add_argument('--eps', type=lambda x: float(x), help = 'value for tolerance to set accuracy of each step for adaptive step size algorithm rkqs, (default = 1e-9)', default = 1e-9)
    args = parser.parse_args()
    main(args.config_path, args.t_start, args.t_final, args.h_0, args.h_min, args.eps)