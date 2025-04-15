# Learning Control Lyapunov Functions with Formal Verification

A data-driven framework for synthesizing Control Lyapunov Functions (CLFs) and stabilizing feedback controllers using formal verification algorithms for nonlinear systems.

## Quick Start

### Step 1: Install dReal (Required)

The package requires dReal for symbolic verification. The easiest way to install it is:

```bash
# Check if dReal is already installed
python install_dreal.py --check

# Install dReal if needed
sudo python install_dreal.py
```

### Step 2: Run the Van der Pol Example

```bash
# Option 1: Run with automatic dReal installation if needed
python run_van_der_pol.py --install

# Option 2: Run the example (if dReal is already installed)
python run_van_der_pol.py
```

This will:
1. Create a Van der Pol oscillator model
2. Train a neural network to learn a Control Lyapunov Function
3. Verify the function using dReal
4. Synthesize a controller using Sontag's formula
5. Simulate the closed-loop system and generate plots

## Installation

### Option 1: Install the package in development mode

```bash
pip install -e control_lyapunov
```

### Option 2: Install from source

```bash
cd control_lyapunov
pip install -e .
```

## Project Structure

```
.
├── install_dreal.py        # dReal installation helper
├── run_van_der_pol.py      # Runner script for Van der Pol example
├── control_lyapunov/       # Main package directory
│   ├── README.md           # Detailed package documentation
│   ├── control_lyapunov/   # Core package modules
│   ├── examples/           # Example implementations
│   ├── tests/              # Unit tests
│   ├── setup.py            # Package installation script
│   └── requirements.txt    # Package dependencies
└── tests/                  # Additional tests
```

## Usage

After installing the package, you can import and use it in your own code:

```python
import control_lyapunov as cl
from control_lyapunov.models.van_der_pol import VanDerPol

# Create a system model
system = VanDerPol(mu=1.0)

# See the package documentation for complete examples
```

## Documentation

For detailed documentation on using the package:
- See the [Package README](control_lyapunov/README.md)
- Check the examples in the [examples directory](control_lyapunov/examples/)

## Requirements

- Python 3.7+
- PyTorch 1.8.0+
- NumPy, SciPy, SymPy
- Matplotlib
- dReal 4.21+

## License

This project is licensed under the MIT License - see the [LICENSE](control_lyapunov/LICENSE) file for details.

