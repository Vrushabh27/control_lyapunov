"""
Complete example demonstrating the use of the control_lyapunov package with the van der Pol oscillator.

This example:
1. Creates a van der Pol oscillator model
2. Trains a neural network to learn a Control Lyapunov Function (CLF)
3. Verifies the CLF using dReal
4. Synthesizes a Sontag controller
5. Simulates the closed-loop system
6. Plots the results
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pkgutil

# Check if dReal is installed
# Note: This block will only work in Jupyter/IPython environments.
# For standalone Python scripts, install dReal manually following README instructions.
try:
    import dreal
except ImportError:
    print("\nERROR: dReal is not installed or missing dependencies.")
    print("The recommended way to install dReal is through a Jupyter notebook:")
    print("\nimport pkgutil")
    print("if not pkgutil.find_loader(\"dreal\"):")
    print("    !curl https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install.sh | bash")
    print("    !pip install dreal --upgrade\n")
    print("Other installation methods may cause errors or missing dependencies.")
    print("See https://github.com/dreal/dreal4 for more information.")
    sys.exit(1)

# Add package root to path if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import package modules
from control_lyapunov.models.van_der_pol import VanDerPol
from control_lyapunov.learning import MultiLayerNet, train_lyapunov_network
from control_lyapunov.verification import create_verifier
from control_lyapunov.sontag import controller_closed_form
from control_lyapunov.simulation import simulate_system, plot_simulation_results, phase_portrait


def main():
    """Main function demonstrating the Control Lyapunov workflow."""
    print("Starting van der Pol Oscillator Control Lyapunov Function Example")
    
    # 1. Create the system model
    print("\n1. Creating van der Pol oscillator model...")
    vdp = VanDerPol(mu=1.0)
    
    # 2. Set up the neural network model
    print("\n2. Setting up neural network model...")
    state_dim = vdp.state_dim
    hidden_layers = [6]  # Single hidden layer with 6 neurons
    output_dim = 1
    lqr_init = vdp.initial_control_weights()
    
    # Create the model
    model = MultiLayerNet(state_dim, hidden_layers, output_dim, lqr_init)
    
    # 3. Configure the training
    print("\n3. Configuring training settings...")
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
    
    # 4. Create a verifier function
    print("\n4. Creating verifier function...")
    verification_bounds = (0.5, 6.0)  # (ball_lb, ball_ub)
    precision = 1e-2
    epsilon = 0  # Strict Lyapunov function (negative definite)
    
    verifier = create_verifier(
        vdp.f_symbolic,
        vdp.g_symbolic,
        None,  # We'll use the learned control
        state_dim,
        verification_bounds,
        precision,
        epsilon
    )
    
    # 5. Train the model with verification
    print("\n5. Training the model with verification...")
    model, loss_history, valid = train_lyapunov_network(
        vdp.system,
        model,
        data_config,
        training_config,
        verifier
    )
    
    # 6. Plot the loss history
    print("\n6. Plotting loss history...")
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Lyapunov Risk')
    plt.yscale('log')
    plt.title('Training Loss History')
    plt.grid(True)
    plt.savefig('loss_history.png')
    
    # Early exit if not valid
    if not valid:
        print("\nWARNING: Could not verify the Lyapunov function!")
        print("Try adjusting parameters and running again.")
        return
    
    print("\nSuccessfully trained and verified a Control Lyapunov Function!")
    
    # 7. Create a Sontag controller
    print("\n7. Creating Sontag controller...")
    controller = controller_closed_form(
        model,
        state_dim,
        vdp.f,
        vdp.g
    )
    
    # 8. Simulate the closed-loop system
    print("\n8. Simulating closed-loop system...")
    x0 = np.array([1.5, 0.0])  # Initial state
    t_span = (0, 10.0)  # 10 seconds simulation
    dt = 0.01  # Time step
    
    t, x, u = simulate_system(
        x0,
        t_span,
        controller,
        vdp.f,
        vdp.g,
        dt
    )
    
    # 9. Plot simulation results
    print("\n9. Plotting simulation results...")
    state_labels = ['$x_1$ (position)', '$x_2$ (velocity)']
    control_labels = ['$u$ (control)']
    
    fig, axes = plot_simulation_results(
        t, x, u,
        state_labels=state_labels,
        control_labels=control_labels,
        title='Controlled van der Pol Oscillator'
    )
    plt.savefig('simulation_results.png')
    
    # 10. Generate phase portrait
    print("\n10. Generating phase portrait...")
    xlim = (-3, 3)
    ylim = (-3, 3)
    
    fig, ax = phase_portrait(
        vdp.f,
        vdp.g,
        controller,
        xlim,
        ylim,
        title='Phase Portrait of Controlled van der Pol Oscillator'
    )
    plt.savefig('phase_portrait.png')
    
    print("\nExample completed successfully! Results saved as PNG files.")
    plt.show()


if __name__ == "__main__":
    main() 