"""
Simulation module for closed-loop systems with Control Lyapunov Functions (CLF).
"""

import numpy as np
from scipy.integrate import solve_ivp
import torch
import matplotlib.pyplot as plt


def closed_loop_dynamics(t, x, controller, system_f, system_g):
    """
    Define the closed-loop dynamics for the controlled system.
    
    Parameters
    ----------
    t : float
        Current time (not used in autonomous systems)
    x : numpy.ndarray
        Current state
    controller : callable
        Controller function that takes state x and returns control input u
    system_f : callable
        Function representing the drift dynamics f(x)
    system_g : callable
        Function representing the control dynamics g(x)
        
    Returns
    -------
    numpy.ndarray
        State derivative dx/dt
    """
    # Compute control input
    u = controller(x)
    
    # Convert to tensor for system functions
    x_tensor = torch.tensor(x, dtype=torch.float32).reshape(1, -1)
    
    # Compute system dynamics
    f_x = system_f(x_tensor).detach().numpy().flatten()
    g_x = system_g(x_tensor).detach().numpy().flatten()
    
    # Compute state derivative
    dx_dt = f_x + g_x * u
    
    return dx_dt


def simulate_system(x0, t_span, controller, system_f, system_g, dt=0.01, method='RK45'):
    """
    Simulate the closed-loop system using scipy's solve_ivp.
    
    Parameters
    ----------
    x0 : numpy.ndarray
        Initial state
    t_span : tuple
        (t_start, t_end) time interval for simulation
    controller : callable
        Controller function that takes state x and returns control input u
    system_f : callable
        Function representing the drift dynamics f(x)
    system_g : callable
        Function representing the control dynamics g(x)
    dt : float, optional
        Time step for simulation output
    method : str, optional
        Integration method ('RK45', 'RK23', etc.)
        
    Returns
    -------
    tuple
        (t, x, u) where t is time array, x is state trajectory array, 
        and u is control input array
    """
    # Create time points for output
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    # Define dynamics function for scipy solver
    dynamics = lambda t, x: closed_loop_dynamics(t, x, controller, system_f, system_g)
    
    # Solve IVP
    sol = solve_ivp(dynamics, t_span, x0, t_eval=t_eval, method=method)
    
    # Extract time and state trajectories
    t = sol.t
    x = sol.y.T
    
    # Compute control inputs for each state
    u = np.zeros((len(t), 1))
    for i in range(len(t)):
        u[i] = controller(x[i])
    
    return t, x, u


def plot_simulation_results(t, x, u=None, state_labels=None, control_labels=None, 
                           title='Closed-Loop Simulation'):
    """
    Plot the simulation results.
    
    Parameters
    ----------
    t : numpy.ndarray
        Time array
    x : numpy.ndarray
        State trajectory array of shape (time_steps, state_dim)
    u : numpy.ndarray, optional
        Control input array of shape (time_steps, control_dim)
    state_labels : list, optional
        Labels for state variables
    control_labels : list, optional
        Labels for control inputs
    title : str, optional
        Title for the plot
        
    Returns
    -------
    tuple
        (fig, axes) matplotlib figure and axes objects
    """
    state_dim = x.shape[1]
    
    if state_labels is None:
        state_labels = [f'x{i+1}' for i in range(state_dim)]
    
    # Create figure
    if u is not None:
        control_dim = u.shape[1]
        if control_labels is None:
            control_labels = [f'u{i+1}' for i in range(control_dim)]
        
        n_rows = state_dim + control_dim
    else:
        n_rows = state_dim
    
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2*n_rows), sharex=True)
    
    if n_rows == 1:
        axes = [axes]
    
    # Plot states
    for i in range(state_dim):
        axes[i].plot(t, x[:, i])
        axes[i].set_ylabel(state_labels[i])
        axes[i].grid(True)
    
    # Plot controls if provided
    if u is not None:
        for i in range(control_dim):
            axes[state_dim + i].plot(t, u[:, i])
            axes[state_dim + i].set_ylabel(control_labels[i])
            axes[state_dim + i].grid(True)
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title)
    plt.tight_layout()
    
    return fig, axes


def phase_portrait(system_f, system_g, controller, xlim, ylim, nx=20, ny=20, 
                  title='Phase Portrait'):
    """
    Generate a phase portrait for a 2D system.
    
    Parameters
    ----------
    system_f : callable
        Function representing the drift dynamics f(x)
    system_g : callable
        Function representing the control dynamics g(x)
    controller : callable
        Controller function that takes state x and returns control input u
    xlim : tuple
        (xmin, xmax) limits for x1 axis
    ylim : tuple
        (ymin, ymax) limits for x2 axis
    nx, ny : int, optional
        Number of grid points in each dimension
    title : str, optional
        Title for the plot
        
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axis objects
    """
    # Only for 2D systems
    x1 = np.linspace(xlim[0], xlim[1], nx)
    x2 = np.linspace(ylim[0], ylim[1], ny)
    
    X1, X2 = np.meshgrid(x1, x2)
    
    DX1 = np.zeros_like(X1)
    DX2 = np.zeros_like(X2)
    
    # Compute vector field
    for i in range(nx):
        for j in range(ny):
            x = np.array([X1[j, i], X2[j, i]])
            dx = closed_loop_dynamics(0, x, controller, system_f, system_g)
            DX1[j, i] = dx[0]
            DX2[j, i] = dx[1]
    
    # Normalize vectors for better visualization
    magnitude = np.sqrt(DX1**2 + DX2**2)
    max_mag = np.max(magnitude)
    
    if max_mag > 0:
        DX1 = DX1 / max_mag
        DX2 = DX2 / max_mag
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.quiver(X1, X2, DX1, DX2)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.grid(True)
    
    return fig, ax 