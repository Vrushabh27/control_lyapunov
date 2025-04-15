"""
Van der Pol oscillator model.

The controlled van der Pol system is:
    x1_dot = x2,
    x2_dot = μ*(1 - x1^2)*x2 - x1 + u,
with μ = 1 by default.
"""

import torch
from dreal import Variable, Expression


class VanDerPol:
    """
    Implementation of the van der Pol oscillator system.
    
    Parameters
    ----------
    mu : float, optional
        System parameter, default is 1.0
    """
    
    def __init__(self, mu=1.0):
        self.mu = mu
        self.state_dim = 2
        self.control_dim = 1
        
    def f(self, x):
        """
        Drift dynamics: f(x).
        
        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch_size, state_dim)
            
        Returns
        -------
        torch.Tensor
            f(x) of shape (batch_size, state_dim)
        """
        batch_size = x.shape[0]
        f_x = torch.zeros((batch_size, self.state_dim), dtype=x.dtype, device=x.device)
        
        f_x[:, 0] = x[:, 1]
        f_x[:, 1] = self.mu * (1 - x[:, 0]**2) * x[:, 1] - x[:, 0]
        
        return f_x
    
    def g(self, x):
        """
        Control dynamics: g(x).
        
        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch_size, state_dim)
            
        Returns
        -------
        torch.Tensor
            g(x) of shape (batch_size, state_dim)
        """
        batch_size = x.shape[0]
        g_x = torch.zeros((batch_size, self.state_dim), dtype=x.dtype, device=x.device)
        
        g_x[:, 1] = 1.0  # Control only affects x2_dot
        
        return g_x
    
    def system(self, x, u):
        """
        Combined system dynamics: f(x) + g(x)*u.
        
        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch_size, state_dim)
        u : torch.Tensor
            Control input tensor of shape (batch_size, 1)
            
        Returns
        -------
        torch.Tensor
            State derivative of shape (batch_size, state_dim)
        """
        f_x = self.f(x)
        g_x = self.g(x)
        
        # Expand control input along state dimension
        u_expanded = u.repeat(1, self.state_dim)
        
        # Element-wise multiplication of g(x) and u
        gu = g_x * u_expanded
        
        return f_x + gu
    
    def f_symbolic(self, x_vars):
        """
        Symbolic drift dynamics for dReal verification.
        
        Parameters
        ----------
        x_vars : list
            List of dReal Variable objects representing the state
            
        Returns
        -------
        list
            List of dReal Expression objects representing f(x)
        """
        x1, x2 = x_vars
        
        # f(x) = [x2, mu*(1-x1^2)*x2 - x1]
        return [x2, self.mu * (1 - x1**2) * x2 - x1]
    
    def g_symbolic(self, x_vars):
        """
        Symbolic control dynamics for dReal verification.
        
        Parameters
        ----------
        x_vars : list
            List of dReal Variable objects representing the state
            
        Returns
        -------
        list
            List of dReal Expression objects representing g(x)
        """
        # g(x) = [0, 1]
        return [Expression(0), Expression(1)]
    
    def initial_control_weights(self):
        """
        Initial LQR-like weights for the control branch.
        
        Returns
        -------
        torch.Tensor
            Initial weight tensor for the control branch
        """
        return torch.tensor([[-2.0, -2.0]])  # Simple linear control law 