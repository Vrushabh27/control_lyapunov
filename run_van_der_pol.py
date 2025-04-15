#!/usr/bin/env python3
"""
Van der Pol Example Runner

This script checks for dReal installation and then runs the van der Pol oscillator example
from the control_lyapunov package. It demonstrates learning a Control Lyapunov Function (CLF)
for the van der Pol system and verifying it with dReal.

Usage:
  python run_van_der_pol.py            # Run the example
  python run_van_der_pol.py --install  # Install dReal and run the example
  python run_van_der_pol.py --help     # Show this help message
"""

import os
import sys
import argparse
import platform
import subprocess
import time

def is_dreal_installed():
    """Check if dReal is installed and functioning."""
    try:
        import dreal
        # Try to create a simple dReal expression to verify it's working
        x = dreal.Variable("x")
        e = dreal.sin(x)
        return True
    except (ImportError, AttributeError):
        return False

def run_example():
    """Run the van der Pol example."""
    print("\n=== Running van der Pol example ===\n")
    
    # Add the current directory to the Python path to ensure we can import the package
    sys.path.append(os.getcwd())
    
    try:
        # Import and run the example
        from control_lyapunov.examples.van_der_pol_example import main
        main()
        return True
    except Exception as e:
        print(f"Error running example: {e}")
        return False

def main():
    """Main function to handle example running and dReal installation."""
    parser = argparse.ArgumentParser(description='Van der Pol Example Runner')
    parser.add_argument('--install', action='store_true', 
                        help='Also install dReal if needed')
    args = parser.parse_args()
    
    print("=== Van der Pol Example Runner ===")
    
    # Check if dReal is installed and working
    if not is_dreal_installed():
        print("dReal is not installed or not working correctly.")
        
        if args.install:
            print("\nInstalling dReal using the installation helper...")
            # Run the dedicated dReal install script
            install_script = os.path.join(os.getcwd(), "install_dreal.py")
            
            if os.path.exists(install_script):
                # Run installation script
                result = subprocess.run([sys.executable, install_script])
                
                if result.returncode != 0:
                    print("\nAutomatic installation failed.")
                    print("Try to install dReal manually:")
                    print("1. sudo apt-get update")
                    print("2. sudo apt-get install -y software-properties-common")
                    print("3. sudo add-apt-repository -y ppa:dreal/dreal")
                    print("4. sudo apt-get update")
                    print("5. sudo apt-get install -y libdreal-dev")
                    print("6. pip install dreal")
                    sys.exit(1)
                
                # Check again if dReal is now properly installed
                if not is_dreal_installed():
                    print("\ndReal installation didn't seem to work correctly.")
                    print("Please try installing manually.")
                    sys.exit(1)
            else:
                print(f"Error: Installation script not found at {install_script}")
                sys.exit(1)
        else:
            print("\ndReal is required to run this example.")
            print("Please run with --install flag to install dReal:")
            print(f"  python {sys.argv[0]} --install")
            print("\nOr install dReal manually following the instructions in the README.")
            sys.exit(1)
    else:
        print("dReal is already installed and working.")
    
    # Run the example
    success = run_example()
    
    if success:
        print("\n=== Example completed successfully! ===")
        print("Check the generated PNG files in the current directory:")
        print("- loss_history.png")
        print("- simulation_results.png")
        print("- phase_portrait.png")
    else:
        print("\n=== Example failed to run correctly. ===")
        print("Please check the error messages above for more information.")

if __name__ == "__main__":
    main() 