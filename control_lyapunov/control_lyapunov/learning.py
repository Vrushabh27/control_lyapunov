"""
Neural network-based Control Lyapunov Function (CLF) learning module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from .utils import compute_lie_derivative, state_norm, generate_training_data


class MultiLayerNet(nn.Module):
    """
    Multi-layer neural network model for representing a candidate
    Lyapunov function and control law.
    
    Parameters
    ----------
    input_dim : int
        Input dimension (state dimension)
    hidden_layers : list
        List with the number of neurons in each hidden layer
    output_dim : int
        Output dimension (1 for candidate V)
    lqr_init : torch.Tensor
        Tensor used to initialize the linear control branch
    """
    
    def __init__(self, input_dim, hidden_layers, output_dim, lqr_init=None):
        super(MultiLayerNet, self).__init__()
        
        self.hidden_layers = nn.ModuleList()
        
        current_dim = input_dim
        for h_dim in hidden_layers:
            self.hidden_layers.append(nn.Linear(current_dim, h_dim))
            current_dim = h_dim
            
        self.output_layer = nn.Linear(current_dim, output_dim)  # Candidate V(x)
        self.control = nn.Linear(input_dim, 1, bias=False)   # Control branch uses original state
        
        # Initialize control weights if provided
        if lqr_init is not None:
            self.control.weight = nn.Parameter(lqr_init)
            
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
            
        Returns
        -------
        tuple
            (V, u) where V is the candidate Lyapunov function value
            and u is the control input
        """
        x_orig = x.clone()  # Save original input for the control branch
        
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
            
        V = torch.tanh(self.output_layer(x))
        u = self.control(x_orig)
        
        return V, u


def train_lyapunov_network(system, model, data_config, training_config, verification=None):
    """
    Train a neural network to learn a Control Lyapunov Function (CLF)
    for a given nonlinear system.
    
    Parameters
    ----------
    system : callable
        Function that takes state x and control u, and returns the dynamics f(x, u)
    model : MultiLayerNet
        Neural network model to be trained
    data_config : dict
        Configuration for training data generation:
        - num_samples: number of training samples
        - state_dim: dimension of the state space
        - bounds: tuple with (lower, upper) bounds for sampling
    training_config : dict
        Configuration for training:
        - learning_rate: learning rate for optimizer
        - max_iterations: maximum number of training iterations
        - optimizer: 'adam' or 'sgd'
    verification : callable, optional
        Function to verify Lyapunov conditions, returns None if verified or counterexamples
        
    Returns
    -------
    tuple
        (model, loss_history, valid) where valid is a boolean indicating
        whether the final model satisfies Lyapunov conditions
    """
    # Set up data and training parameters
    num_samples = data_config.get('num_samples', 500)
    state_dim = data_config.get('state_dim', 2)
    bounds = data_config.get('bounds', (-6, 6))
    
    learning_rate = training_config.get('learning_rate', 0.01)
    max_iterations = training_config.get('max_iterations', 2000)
    optimizer_type = training_config.get('optimizer', 'adam')
    
    # Generate initial training data
    x = generate_training_data(state_dim, num_samples, bounds)
    x_0 = torch.zeros([1, state_dim]).float()
    
    # Set up optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    # Training loop
    loss_history = []
    iteration = 0
    valid = False
    
    start_time = time.time()
    verification_time = 0
    
    while iteration < max_iterations and not valid:
        # Forward pass
        V_candidate, u = model(x)
        X0, _ = model(x_0)
        
        # Compute dynamics with control input
        f_dyn = system(x, u)
        
        # Compute Lie derivative
        L_V = compute_lie_derivative(model, x, f_dyn)
        
        # Compute circle tuning term
        Circle_Tuning = state_norm(x)
        
        # Compute Lyapunov risk
        Lyapunov_risk = (F.relu(-V_candidate) + 1.5*F.relu(L_V+0.5)).mean() \
                      + 2.2*((Circle_Tuning - 6*V_candidate).pow(2)).mean() \
                      + (X0).pow(2)
        
        print(f"{iteration}, Lyapunov Risk = {Lyapunov_risk.item()}")
        loss_history.append(Lyapunov_risk.item())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        Lyapunov_risk.backward()
        optimizer.step()
        
        # Verification (if provided)
        if verification is not None and iteration % 10 == 0:
            print('=========== Verifying ==========')
            start_verif = time.time()
            result = verification(model)
            end_verif = time.time()
            verification_time += (end_verif - start_verif)
            
            if result is None:
                valid = True
                print("Satisfies conditions!!")
            else:
                print("Not a Lyapunov function. Found counterexample.")
                # Add counterexamples to training data
                x = torch.cat([x, result], dim=0)
                
            print('================================')
        
        iteration += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f'\nTotal training time: {total_time}')
    print(f"Total verification time: {verification_time}")
    
    return model, loss_history, valid 