"""
Sontag controller synthesis module for Control Lyapunov Functions (CLF).
"""

import torch
import numpy as np
from .utils import compute_gradient


def sontag_controller(model, x, system_f, system_g, threshold=1e-6):
    """
    Compute the Sontag universal formula controller for a given system and CLF.
    
    The Sontag controller is defined as:
    u(x) = - (L_fV + sqrt(L_fV^2 + L_gV^4)) / L_gV  if |L_gV| > threshold, otherwise 0.
    
    Where:
    L_fV = ∇V(x)·f(x)
    L_gV = ∇V(x)·g(x)
    
    Parameters
    ----------
    model : torch.nn.Module
        Neural network model representing the Lyapunov function
    x : torch.Tensor
        State tensor of shape (batch_size, state_dim)
    system_f : callable
        Function representing the drift dynamics f(x)
    system_g : callable
        Function representing the control dynamics g(x)
    threshold : float, optional
        Threshold below which L_gV is considered zero
        
    Returns
    -------
    torch.Tensor
        Control input tensor of shape (batch_size, 1)
    """
    # Compute gradient of V
    grad_V = compute_gradient(model, x)
    
    # Compute f(x) and g(x)
    f_x = system_f(x)
    g_x = system_g(x)
    
    # Compute Lie derivatives
    LfV = torch.sum(grad_V * f_x, dim=1, keepdim=True)
    LgV = torch.sum(grad_V * g_x, dim=1, keepdim=True)
    
    # Apply Sontag's formula with regularization for small LgV
    u = torch.where(
        torch.abs(LgV) > threshold,
        -(LfV + torch.sqrt(LfV**2 + LgV**4)) / LgV,
        torch.zeros_like(LgV)
    )
    
    return u


def controller_closed_form(model, state_dim, system_f, system_g):
    """
    Return a closed-form Sontag controller function for a given system and CLF.
    
    Parameters
    ----------
    model : torch.nn.Module
        Neural network model representing the Lyapunov function
    state_dim : int
        Dimension of the state space
    system_f : callable
        Function representing the drift dynamics f(x)
    system_g : callable
        Function representing the control dynamics g(x)
        
    Returns
    -------
    callable
        Function that takes a state vector and returns a control input
    """
    def controller(x_np):
        """
        Compute control input for a given state.
        
        Parameters
        ----------
        x_np : numpy.ndarray
            State vector of shape (state_dim,)
            
        Returns
        -------
        numpy.ndarray
            Control input vector
        """
        x_tensor = torch.tensor(x_np, dtype=torch.float32).reshape(1, -1)
        u_tensor = sontag_controller(model, x_tensor, system_f, system_g)
        return u_tensor.detach().numpy().flatten()
    
    return controller


def lyapunov_barrier_controller(model, x, system_f, system_g, barrier_function=None, 
                               barrier_weight=1.0, threshold=1e-6):
    """
    Compute a control law using both a Lyapunov function and a barrier function.
    
    Parameters
    ----------
    model : torch.nn.Module
        Neural network model representing the Lyapunov function
    x : torch.Tensor
        State tensor of shape (batch_size, state_dim)
    system_f : callable
        Function representing the drift dynamics f(x)
    system_g : callable
        Function representing the control dynamics g(x)
    barrier_function : callable, optional
        Function h(x) where h(x) > 0 defines the safe region
    barrier_weight : float, optional
        Weight for barrier term in combined controller
    threshold : float, optional
        Threshold below which derivatives are considered zero
        
    Returns
    -------
    torch.Tensor
        Control input tensor of shape (batch_size, 1)
    """
    # Basic Sontag controller from Lyapunov function
    u_lyap = sontag_controller(model, x, system_f, system_g, threshold)
    
    # If no barrier function, return standard controller
    if barrier_function is None:
        return u_lyap
    
    # TODO: Implement barrier function controller
    # This would combine the Lyapunov controller with a barrier
    # function to ensure safety constraints
    
    return u_lyap  # For now, just return Lyapunov controller 