"""
Inverted pendulum on a cart model.

The controlled inverted pendulum system is:
    x1_dot = x2,
    x2_dot = (g*sin(x1) - u*m*l*cos(x1)) / (m*l^2),
where:
    x1: angle from the upright position
    x2: angular velocity
    g: gravitational acceleration
    m: mass of the pendulum
    l: length of the pendulum
    u: control input (horizontal force applied to the cart)
"""

import torch
import numpy as np
from dreal import Variable, Expression, sin, cos


class InvertedPendulum:
    """
    Implementation of the inverted pendulum system.
    
    Parameters
    ----------
    m : float, optional
        Mass of the pendulum, default is 1.0 kg
    l : float, optional
        Length of the pendulum, default is 1.0 m
    g : float, optional
        Gravitational acceleration, default is 9.81 m/s^2
    """
    
    def __init__(self, m=1.0, l=1.0, g=9.81):
        self.m = m  # mass (kg)
        self.l = l  # length (m)
        self.g = g  # gravity (m/s^2)
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
        f_x[:, 1] = self.g * torch.sin(x[:, 0]) / (self.m * self.l**2)
        
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
        
        g_x[:, 1] = -torch.cos(x[:, 0]) / (self.m * self.l)
        
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
        
        # f(x) = [x2, g*sin(x1)/(m*l^2)]
        return [x2, self.g * sin(x1) / (self.m * self.l**2)]
    
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
        x1, _ = x_vars
        
        # g(x) = [0, -cos(x1)/(m*l)]
        return [Expression(0), -cos(x1) / (self.m * self.l)]
    
    def initial_control_weights(self):
        """
        Initial LQR-like weights for the control branch.
        
        Returns
        -------
        torch.Tensor
            Initial weight tensor for the control branch
        """
        # Simple linear state feedback for initial control
        return torch.tensor([[5.0, 2.0]])
    
    def linearized_matrices(self):
        """
        Return the linearized system matrices around the upright position.
        
        Returns
        -------
        tuple
            (A, B) matrices for the linearized system
        """
        # Linearization around x = [0, 0]
        A = np.array([
            [0, 1],
            [self.g/(self.m * self.l**2), 0]
        ])
        
        B = np.array([
            [0],
            [-1/(self.m * self.l)]
        ])
        
        return A, B 