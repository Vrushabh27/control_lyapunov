#!/usr/bin/env python3
"""
Package Build and Upload Script

This script builds the control-lyapunov package and uploads it to PyPI.
You must have twine installed: pip install twine

Usage:
  python build_package.py --build     # Just build the package
  python build_package.py --upload    # Build and upload to PyPI
  python build_package.py --test      # Build and upload to TestPyPI
"""

import os
import subprocess
import sys
import shutil
import argparse

def clean_build_dirs():
    """Remove old build artifacts."""
    print("🧹 Cleaning build directories...")
    dirs_to_clean = ['dist', 'build', '*.egg-info']
    
    for pattern in dirs_to_clean:
        for path in os.path.join('control_lyapunov', pattern), pattern:
            if '*' in path:
                # Use find for patterns
                os.system(f"find . -name '{pattern}' -type d -exec rm -rf {{}} +")
            elif os.path.exists(path):
                # Remove directory
                shutil.rmtree(path)
    
    print("✅ Build directories cleaned")

def build_package():
    """Build the Python package."""
    print("\n📦 Building package...")
    os.chdir('control_lyapunov')
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "build"], check=True)
    subprocess.run([sys.executable, "-m", "build"], check=True)
    print("✅ Package built successfully")
    os.chdir('..')

def upload_to_pypi(test=False):
    """Upload the built package to PyPI or TestPyPI."""
    repository = "--repository testpypi" if test else ""
    repo_name = "TestPyPI" if test else "PyPI"
    
    print(f"\n🚀 Uploading package to {repo_name}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "twine"], check=True)
    
    # Find all distribution files
    dist_files = os.path.join('control_lyapunov', 'dist', '*')
    
    if test:
        cmd = f"twine upload --repository testpypi {dist_files}"
    else:
        cmd = f"twine upload {dist_files}"
        
    print(f"Running: {cmd}")
    os.system(cmd)
    print(f"✅ Package uploaded to {repo_name}")
    
    if test:
        print("\n📋 To install from TestPyPI:")
        print("pip install --index-url https://test.pypi.org/simple/ control-lyapunov")
    else:
        print("\n📋 To install from PyPI:")
        print("pip install control-lyapunov")

def main():
    """Parse arguments and execute requested actions."""
    parser = argparse.ArgumentParser(description="Build and publish the control-lyapunov package")
    parser.add_argument("--build", action="store_true", help="Build the package")
    parser.add_argument("--upload", action="store_true", help="Upload to PyPI")
    parser.add_argument("--test", action="store_true", help="Upload to TestPyPI")
    
    args = parser.parse_args()
    
    if not (args.build or args.upload or args.test):
        parser.print_help()
        return
    
    # Always clean first
    clean_build_dirs()
    
    # Build if requested or needed for upload
    if args.build or args.upload or args.test:
        build_package()
    
    # Upload if requested
    if args.upload:
        upload_to_pypi()
    elif args.test:
        upload_to_pypi(test=True)
    
    print("\n🎉 All operations completed successfully!")

if __name__ == "__main__":
    main() 