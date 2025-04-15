# Control Lyapunov Functions (CLF) Package

A data-driven framework for synthesizing feedback controllers using learned Control Lyapunov Functions (CLFs) for nonlinear systems.

Developed by Vrushabh Zinage and Shrenik Zinage.

## Problem Overview

Designing feedback-stabilizing controllers for nonlinear systems is often analytically intractable. This package implements a machine learning approach to synthesize stabilizing controllers for nonlinear control-affine systems of the form:

```
ẋ = f(x) + g(x)u
```

where x ∈ ℝⁿ, u ∈ ℝᵐ, and f(x), g(x) are nonlinear vector fields.

The package workflow:

1. **Learning**: Uses a neural network to represent a candidate Lyapunov function V(x)
2. **Verification**: Verifies V(x) using symbolic reasoning via the dReal solver
3. **Synthesis**: Synthesizes a stabilizing controller using Sontag's universal formula
4. **Simulation**: Simulates the closed-loop system using RK45 integration

## Installation

### Step 1: Installing dReal (REQUIRED)

The package relies on dReal for symbolic verification of Lyapunov functions. Installation instructions for Ubuntu:

#### Ubuntu (Recommended Method)

```bash
# Add the dReal PPA repository
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:dreal/dreal
sudo apt-get update

# Install dReal with dependencies
sudo apt-get install -y libdreal-dev

# Install the Python package
pip install dreal
```

You can verify the installation by running:

```python
import dreal
x = dreal.Variable("x")
print(dreal.sin(x))  # Should print sin(x) without errors
```

#### Troubleshooting dReal Installation

If you encounter any issues:

1. Make sure all dependencies are installed:
   ```bash
   sudo apt-get install -y build-essential pkg-config libgmp-dev
   ```

2. Check for any error messages when importing dReal:
   ```python
   import dreal
   ```

3. If you see errors about missing libraries, you may need to install:
   ```bash
   sudo apt-get install -y libstdc++6
   ```

4. For other Linux distributions or platforms, refer to the [dReal documentation](https://github.com/dreal/dreal4).

### Step 2: Install the Control Lyapunov Package

#### Option 1: Install from PyPI

```bash
pip install control_lyapunov
```

#### Option 2: Install from source

```bash
git clone https://github.com/Vrushabh27/control_lyapunov.git
cd control_lyapunov
pip install -e .
```

## Usage

### 1. Define Your Nonlinear System

Create a custom system by implementing the required functions, or use one of the predefined models:

```python
from control_lyapunov.models.van_der_pol import VanDerPol

# Create an instance of the van der Pol oscillator
system = VanDerPol(mu=1.0)  # mu is the system parameter
```

For a custom system, implement a class with the following methods:
- `f(x)`: Drift dynamics
- `g(x)`: Control dynamics
- `system(x, u)`: Combined system dynamics
- `f_symbolic(x_vars)`: Symbolic drift dynamics for verification
- `g_symbolic(x_vars)`: Symbolic control dynamics for verification

### 2. Create and Train a Lyapunov Function

```python
from control_lyapunov.learning import MultiLayerNet, train_lyapunov_network

# Define neural network architecture
state_dim = system.state_dim
hidden_layers = [6]  # Single hidden layer with 6 neurons
output_dim = 1  # Scalar Lyapunov function
lqr_init = system.initial_control_weights()

# Create the model
model = MultiLayerNet(state_dim, hidden_layers, output_dim, lqr_init)

# Configure training
data_config = {
    'num_samples': 500,
    'state_dim': state_dim,
    'bounds': (-6, 6)
}

training_config = {
    'learning_rate': 0.01,
    'max_iterations': 2000,
    'optimizer': 'adam'
}

# Create a verifier
from control_lyapunov.verification import create_verifier

verifier = create_verifier(
    system.f_symbolic,
    system.g_symbolic,
    None,  # We'll use the learned control
    state_dim,
    (0.5, 6.0),  # (ball_lb, ball_ub) for verification
    1e-2,  # precision
    0  # epsilon (0 for strict Lyapunov)
)

# Train the model
model, loss_history, valid = train_lyapunov_network(
    system.system,
    model,
    data_config,
    training_config,
    verifier
)
```

### 3. Synthesize a Controller

```python
from control_lyapunov.sontag import controller_closed_form

# Create a Sontag controller
controller = controller_closed_form(
    model,
    state_dim,
    system.f,
    system.g
)
```

### 4. Simulate the Closed-Loop System

```python
from control_lyapunov.simulation import simulate_system, plot_simulation_results

# Define initial state and simulation parameters
x0 = np.array([1.5, 0.0])  # Initial state
t_span = (0, 10.0)  # 10 seconds simulation
dt = 0.01  # Time step

# Simulate
t, x, u = simulate_system(
    x0,
    t_span,
    controller,
    system.f,
    system.g,
    dt
)

# Plot results
state_labels = ['$x_1$ (position)', '$x_2$ (velocity)']
control_labels = ['$u$ (control)']

fig, axes = plot_simulation_results(
    t, x, u,
    state_labels=state_labels, 
    control_labels=control_labels,
    title='Controlled System'
)
```

## Complete Example

See the complete example for the van der Pol oscillator:

```python
import control_lyapunov as cl
from control_lyapunov.models.van_der_pol import VanDerPol

# Create the system
vdp = VanDerPol(mu=1.0)

# Create and train the model
# ... (see examples/van_der_pol_example.py for the complete workflow)
```

The `examples` directory contains full examples for different systems.

## Running Examples

To run the Van der Pol example:

```bash
# Make sure dReal is installed first
python -m control_lyapunov.examples.van_der_pol_example
```

## Configuration Options

The package offers numerous tuning options:

### Neural Network Configuration
- Number of hidden layers and neurons
- Activation functions
- Initial control weights

### Training Configuration
- Learning rate
- Optimizer (Adam, SGD)
- Number of iterations
- Training data size and sampling bounds

### Verification Configuration
- Verification region (ball_lb, ball_ub)
- Precision for dReal solver
- Epsilon value for Lie derivative bound

### Simulation Configuration
- Integration method
- Time step
- Simulation duration
- Initial state

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* This package was developed based on research in neural Lyapunov functions and data-driven control.
* Special thanks to the dReal team for their symbolic reasoning tool.

## Citation

If you use this package in your research, please cite:

```
@software{control_lyapunov,
  author = {Zinage, Vrushabh and Zinage, Shrenik},
  title = {Control Lyapunov Functions (CLF) Package},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/Vrushabh27/control_lyapunov}
}
``` 