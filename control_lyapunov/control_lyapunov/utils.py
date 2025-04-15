"""
Utility functions for Control Lyapunov Functions (CLF) package.
"""

import torch
import numpy as np


def compute_gradient(model, x):
    """
    Compute the gradient of the candidate Lyapunov function V(x) with respect to x.
    
    Parameters
    ----------
    model : torch.nn.Module
        Neural network model representing the Lyapunov function
    x : torch.Tensor
        Input tensor of shape (batch_size, state_dim)
        
    Returns
    -------
    torch.Tensor
        Gradient of V with respect to x, shape (batch_size, state_dim)
    """
    x = x.clone().detach().requires_grad_(True)
    V, _ = model(x)
    grad_V = torch.autograd.grad(outputs=V, inputs=x,
                                grad_outputs=torch.ones_like(V),
                                create_graph=True)[0]
    return grad_V


def compute_lie_derivative(model, x, f):
    """
    Compute the Lie derivative L_V(x) = ∇V(x) · f(x,u) using autograd.
    
    Parameters
    ----------
    model : torch.nn.Module
        Neural network model representing the Lyapunov function
    x : torch.Tensor
        Input tensor of shape (batch_size, state_dim)
    f : torch.Tensor
        Vector field evaluated at x, shape (batch_size, state_dim)
        
    Returns
    -------
    torch.Tensor
        Lie derivative of V along f, shape (batch_size,)
    """
    grad_V = compute_gradient(model, x)
    L_V = (grad_V * f).sum(dim=1)
    return L_V


def state_norm(x):
    """
    Compute the squared norm of state vectors.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, state_dim)
        
    Returns
    -------
    torch.Tensor
        Tensor of shape (batch_size, 1) containing the norm of each state vector
    """
    y = []
    for r in range(len(x)):
        v = sum(x[r][j]**2 for j in range(x.shape[1]))
        y.append([torch.sqrt(v)])
    return torch.tensor(y)


def generate_training_data(state_dim, num_samples, bounds=(-6, 6)):
    """
    Generate random training data within specified bounds.
    
    Parameters
    ----------
    state_dim : int
        Dimension of the state space
    num_samples : int
        Number of samples to generate
    bounds : tuple
        (lower_bound, upper_bound) for uniform sampling
        
    Returns
    -------
    torch.Tensor
        Tensor of shape (num_samples, state_dim) containing random states
    """
    return torch.Tensor(num_samples, state_dim).uniform_(bounds[0], bounds[1]).float() 