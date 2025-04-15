"""
Symbolic verification module for Control Lyapunov Functions (CLF) using dReal.
"""

import sympy as sp
from dreal import Variable, Expression, logical_and, logical_imply, logical_not, CheckSatisfiability
from dreal import Config, sin, cos, tan, exp, log, tanh
import torch
import numpy as np


def check_lyapunov_conditions(x_vars, f_sym, V_sym, ball_lb, ball_ub, config, epsilon=0):
    """
    Check the Lyapunov conditions for all x in the annulus defined by ball_lb and ball_ub:
    if x in the ball then V(x) ≥ 0 and the Lie derivative L_V(x) ≤ epsilon.
    Uses dReal to search for a counterexample.
    
    Parameters
    ----------
    x_vars : list
        List of dReal Variable objects representing the state
    f_sym : list
        List of dReal Expression objects representing the dynamics
    V_sym : dReal.Expression
        dReal expression representing the candidate Lyapunov function
    ball_lb : float
        Lower bound of the ball radius
    ball_ub : float
        Upper bound of the ball radius
    config : dReal.Config
        Configuration for dReal
    epsilon : float, optional
        Upper bound for the Lie derivative, default is 0
        
    Returns
    -------
    dReal.Box or None
        Counter-example if found, None otherwise
    """
    # Compute squared norm
    ball = Expression(0)
    for xi in x_vars:
        ball += xi * xi
    
    # Compute Lie derivative
    lie_derivative_of_V = Expression(0)
    for i in range(len(x_vars)):
        lie_derivative_of_V += f_sym[i] * V_sym.Differentiate(x_vars[i])
    
    # Define the conditions
    ball_in_bound = logical_and(ball_lb * ball_lb <= ball, ball <= ball_ub * ball_ub)
    condition = logical_and(logical_imply(ball_in_bound, V_sym >= 0),
                           logical_imply(ball_in_bound, lie_derivative_of_V <= epsilon))
    
    # Check if conditions are not satisfied
    return CheckSatisfiability(logical_not(condition), config)


def add_counterexamples(x, counter_examples, N, state_dim):
    """
    For each counterexample (given as a box in ℝ^D by dReal), sample N new points uniformly and add them to x.
    
    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape (batch_size, state_dim) containing existing data points
    counter_examples : dReal.Box
        Counterexamples found by dReal
    N : int
        Number of samples to generate for each counterexample
    state_dim : int
        Dimension of the state space
        
    Returns
    -------
    torch.Tensor
        Tensor with added counterexamples
    """
    new_points = []
    for i in range(counter_examples.size()):
        lb = counter_examples[i].lb()  # lower bound vector
        ub = counter_examples[i].ub()  # upper bound vector
        samples = np.random.uniform(lb, ub, (N, state_dim))
        new_points.append(samples)
    
    if new_points:
        new_points = np.concatenate(new_points, axis=0)
        new_points_t = torch.tensor(new_points, dtype=torch.float32)
        return torch.cat([x, new_points_t], dim=0)
    else:
        return x


def build_symbolic_V_from_model(model, input_dim):
    """
    Reconstruct a symbolic (Sympy) expression for V(x) from the trained network.
    
    Parameters
    ----------
    model : MultiLayerNet
        Trained neural network model
    input_dim : int
        Input dimension (state dimension)
        
    Returns
    -------
    tuple
        (x_syms, V_sym) where x_syms are the symbols and V_sym the expression
    """
    x_syms = sp.symbols(f'x1:{input_dim+1}')
    weights = {}
    biases = {}
    layer_index = 1
    
    for layer in model.hidden_layers:
        W = layer.weight.detach().numpy()
        b = layer.bias.detach().numpy()
        out_features, in_features = W.shape
        
        for k in range(out_features):
            for j in range(in_features):
                weights[(layer_index, j+1, k+1)] = sp.nsimplify(W[k, j])
            biases[(layer_index, k+1)] = sp.nsimplify(b[k])
        
        layer_index += 1
    
    W = model.output_layer.weight.detach().numpy()
    b = model.output_layer.bias.detach().numpy()
    out_features, in_features = W.shape
    
    for k in range(out_features):
        for j in range(in_features):
            weights[(layer_index, j+1, k+1)] = sp.nsimplify(W[k, j])
        biases[(layer_index, k+1)] = sp.nsimplify(b[k])
    
    def mlp_forward_all_tanh(x_list, weights, biases, num_layers):
        layer_input = list(x_list)
        
        for i in range(1, num_layers+1):
            neurons = sorted([k for (layer, k) in biases.keys() if layer == i])
            layer_output = []
            
            for k in neurons:
                neuron_inp = sum(weights[(i, j, k)] * layer_input[j-1]
                                for j in range(1, len(layer_input)+1)) + biases[(i, k)]
                neuron_out = sp.tanh(neuron_inp)
                layer_output.append(neuron_out)
            
            layer_input = layer_output
        
        return layer_input[0]
    
    num_layers = len(model.hidden_layers) + 1
    V_sym = mlp_forward_all_tanh(x_syms, weights, biases, num_layers)
    
    return x_syms, V_sym


def sympy_to_dreal(expr, var_mapping):
    """
    Recursively converts a Sympy expression to a dReal Expression.
    
    Parameters
    ----------
    expr : sympy.Expr
        Sympy expression to convert
    var_mapping : dict
        Dictionary mapping symbol names (strings) to dReal Variables
        
    Returns
    -------
    dReal.Expression
        Equivalent dReal expression
    """
    if expr.is_Number:
        return Expression(float(expr))
    
    if expr.is_Symbol:
        name = str(expr)
        if name in var_mapping:
            return var_mapping[name]
        else:
            raise ValueError(f"Symbol {name} not in mapping.")
    
    if expr.is_Add:
        res = sympy_to_dreal(expr.args[0], var_mapping)
        for arg in expr.args[1:]:
            res = res + sympy_to_dreal(arg, var_mapping)
        return res
    
    if expr.is_Mul:
        res = sympy_to_dreal(expr.args[0], var_mapping)
        for arg in expr.args[1:]:
            res = res * sympy_to_dreal(arg, var_mapping)
        return res
    
    if expr.is_Pow:
        base = sympy_to_dreal(expr.args[0], var_mapping)
        exp = expr.args[1]
        if exp.is_Number:
            return base ** float(exp)
        else:
            raise NotImplementedError("Non-numeric exponent: " + str(expr))
    
    if expr.func == sp.tanh:
        arg = sympy_to_dreal(expr.args[0], var_mapping)
        return tanh(arg)
    
    if expr.func == sp.sin:
        arg = sympy_to_dreal(expr.args[0], var_mapping)
        return sin(arg)
    
    if expr.func == sp.cos:
        arg = sympy_to_dreal(expr.args[0], var_mapping)
        return cos(arg)
    
    if expr.func == sp.log:
        arg = sympy_to_dreal(expr.args[0], var_mapping)
        return log(arg)
    
    if expr.func == sp.exp:
        arg = sympy_to_dreal(expr.args[0], var_mapping)
        return exp(arg)
    
    if expr.func == sp.sqrt:
        arg = sympy_to_dreal(expr.args[0], var_mapping)
        return arg ** 0.5
    
    raise NotImplementedError("Conversion not implemented for: " + str(expr))


def create_verifier(system_f, system_g, u_formula, state_dim, bounds, precision=1e-2, epsilon=0):
    """
    Create a verification function for a given dynamical system and controller.
    
    Parameters
    ----------
    system_f : callable
        Function representing the drift term f(x)
    system_g : callable
        Function representing the control term g(x)
    u_formula : callable
        Control law u(x)
    state_dim : int
        Dimension of the state space
    bounds : tuple
        (ball_lb, ball_ub) for the verification region
    precision : float, optional
        Precision for dReal solver
    epsilon : float, optional
        Upper bound for the Lie derivative
        
    Returns
    -------
    callable
        Function that takes a model and returns None if verified or counterexamples if falsified
    """
    # Create dReal variables
    x_vars = [Variable(f"x{i+1}") for i in range(state_dim)]
    var_mapping = {f'x{i+1}': x_vars[i] for i in range(state_dim)}
    
    # Create dReal config
    config = Config()
    config.use_polytope_in_forall = True
    config.use_local_optimization = True
    config.precision = precision
    
    ball_lb, ball_ub = bounds
    
    def verifier(model):
        # Get symbolic form of the Lyapunov function
        x_syms, V_sym = build_symbolic_V_from_model(model, state_dim)
        
        # Convert symbolic control
        control_weights = model.control.weight.detach().numpy()[0]
        u_symbolic = sum(control_weights[i] * x_vars[i] for i in range(state_dim))
        
        # Create symbolic system dynamics with control
        x_numpy = np.zeros((1, state_dim))
        for i in range(state_dim):
            x_numpy[0, i] = 1.0  # Just a placeholder value
        
        # Get symbolic dynamics
        f_symbolic = []
        for i in range(state_dim):
            f_i = system_f(x_vars)[i]
            g_i = system_g(x_vars)[i]
            f_symbolic.append(f_i + g_i * u_symbolic)
        
        # Convert Lyapunov function to dReal
        V_exact = sympy_to_dreal(V_sym, var_mapping)
        
        # Check Lyapunov conditions
        result = check_lyapunov_conditions(
            x_vars, f_symbolic, V_exact, ball_lb, ball_ub, config, epsilon
        )
        
        if result:
            # Convert counterexamples to PyTorch tensor
            counterexamples = add_counterexamples(
                torch.tensor([]), result, 10, state_dim
            )
            return counterexamples
        else:
            return None
    
    return verifier 