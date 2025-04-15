"""
Basic test to verify the package installation and imports.
"""

import unittest


class TestImports(unittest.TestCase):
    """Test that all modules can be imported."""
    
    def test_import_main(self):
        """Test importing the main package."""
        try:
            import control_lyapunov
            self.assertEqual(control_lyapunov.__name__, "control_lyapunov")
        except ImportError as e:
            self.fail(f"Failed to import control_lyapunov: {e}")
    
    def test_import_modules(self):
        """Test importing all submodules."""
        modules = [
            "control_lyapunov.learning",
            "control_lyapunov.verification",
            "control_lyapunov.sontag",
            "control_lyapunov.simulation",
            "control_lyapunov.utils",
            "control_lyapunov.models.van_der_pol",
            "control_lyapunov.models.inverted_pendulum",
        ]
        
        for module in modules:
            try:
                __import__(module)
            except ImportError as e:
                self.fail(f"Failed to import {module}: {e}")
    
    def test_version(self):
        """Test that the version is defined."""
        import control_lyapunov
        self.assertIsNotNone(control_lyapunov.__version__)


if __name__ == "__main__":
    unittest.main() 