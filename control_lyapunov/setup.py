"""
setup.py for control_lyapunov package.
This is for backward compatibility with pip, as modern packages
should primarily use pyproject.toml.
"""

from setuptools import setup, find_packages

setup(
    name="control-lyapunov",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "scipy>=1.6.0",
        "sympy>=1.7.0",
        "matplotlib>=3.3.0",
    ],
    author="Vrushabh Zinage, Shrenik Zinage",
    author_email="vrushabh@example.com",
    description="A data-driven framework for synthesizing feedback controllers using learned Control Lyapunov Functions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Vrushabh27/control-lyapunov",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
) 