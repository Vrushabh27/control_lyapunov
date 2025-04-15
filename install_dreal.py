#!/usr/bin/env python3
"""
dReal Installation Helper Script

This script automates the installation of dReal and its dependencies on Ubuntu.
It provides a command-line interface with various options for installation.

Usage:
  python install_dreal.py             # Install dReal
  python install_dreal.py --check     # Check if dReal is already installed
  python install_dreal.py --help      # Show help information
"""

import sys
import os
import subprocess
import argparse
import platform


def check_dreal_installed():
    """Check if dReal is properly installed and working."""
    try:
        import dreal
        # Create a simple dReal expression to verify functionality
        x = dreal.Variable("x")
        e = dreal.sin(x)
        print("✅ dReal is properly installed and working!")
        return True
    except ImportError:
        print("❌ dReal is not installed.")
        return False
    except Exception as e:
        print(f"❌ dReal is installed but encountered an error: {e}")
        return False


def install_system_dependencies():
    """Install dReal system dependencies."""
    print("\n📦 Installing dReal system dependencies...")
    
    # Check if we're on Ubuntu/Debian
    if platform.system() != "Linux" or not os.path.exists("/etc/apt"):
        print("⚠️  This script is designed for Ubuntu/Debian Linux.")
        print("For other platforms, please follow manual installation at:")
        print("https://github.com/dreal/dreal4")
        return False
    
    # Check if running with sudo
    if os.geteuid() != 0:
        print("⚠️  System dependencies installation requires sudo privileges.")
        print(f"Please run: sudo {sys.executable} {' '.join(sys.argv)}")
        return False
    
    commands = [
        ["apt-get", "update"],
        ["apt-get", "install", "-y", "software-properties-common"],
        ["add-apt-repository", "ppa:dreal/dreal", "-y"],
        ["apt-get", "update"],
        ["apt-get", "install", "-y", "libdreal-dev"]
    ]
    
    try:
        for cmd in commands:
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        
        print("✅ System dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing system dependencies: {e}")
        return False


def install_python_package():
    """Install dReal Python package."""
    print("\n📦 Installing dReal Python package...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "dreal", "--upgrade"], 
                      check=True)
        print("✅ dReal Python package installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dReal Python package: {e}")
        return False


def main():
    """Main function to handle dReal installation."""
    parser = argparse.ArgumentParser(description="dReal Installation Helper")
    parser.add_argument("--check", action="store_true", 
                       help="Check if dReal is already installed")
    parser.add_argument("--skip-deps", action="store_true",
                       help="Skip system dependencies installation")
    
    args = parser.parse_args()
    
    print("=== dReal Installation Helper ===")
    
    # Only check installation if requested
    if args.check:
        check_dreal_installed()
        return
    
    # Check if dReal is already installed
    if check_dreal_installed():
        print("dReal is already installed. No action needed.")
        return
    
    # Install system dependencies if not skipped
    if not args.skip_deps:
        if not install_system_dependencies():
            print("\n⚠️  Failed to install system dependencies.")
            print("You may need to install them manually:")
            print("  sudo apt-get update")
            print("  sudo apt-get install -y software-properties-common")
            print("  sudo add-apt-repository ppa:dreal/dreal -y")
            print("  sudo apt-get update")
            print("  sudo apt-get install -y libdreal-dev")
            if not os.geteuid() == 0:
                print("\nOr try running this script with sudo:")
                print(f"  sudo {sys.executable} {' '.join(sys.argv)}")
            return
    
    # Install Python package
    if not install_python_package():
        print("\n⚠️  Failed to install dReal Python package.")
        print("Try installing it manually: pip install dreal")
        return
    
    # Verify installation
    print("\n🔍 Verifying dReal installation...")
    if check_dreal_installed():
        print("\n🎉 dReal installation completed successfully!")
        print("\nYou can now use dReal in your Python code:")
        print("  import dreal")
        print("  x = dreal.Variable('x')")
        print("  print(dreal.sin(x))")
    else:
        print("\n⚠️  dReal installation verification failed.")
        print("Please check the error messages above.")
        print("You may need to install additional dependencies or refer to:")
        print("https://github.com/dreal/dreal4")


if __name__ == "__main__":
    main() 