"""
Unit tests for the package validator.
"""
import unittest
import tempfile
import json
import os
from unittest.mock import patch, MagicMock

# Import the module to test
from archiveasy.security.package_validator import PackageValidator, create_validator

class TestPackageValidator(unittest.TestCase):
    """Test the PackageValidator class."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a temporary whitelist file
        self.temp_whitelist = tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False)
        self.temp_whitelist.write(json.dumps({
            "requests": {"purpose": "HTTP library", "pypi_url": "https://pypi.org/project/requests/"},
            "pandas": {"purpose": "Data analysis", "pypi_url": "https://pypi.org/project/pandas/"},
            "numpy": {"purpose": "Scientific computing", "pypi_url": "https://pypi.org/project/numpy/"}
        }))
        self.temp_whitelist.close()
        
        # Create validators
        self.whitelist_validator = PackageValidator(
            whitelist_path=self.temp_whitelist.name,
            mode="whitelist"
        )
        
        self.verify_validator = PackageValidator(
            whitelist_path=self.temp_whitelist.name,
            mode="verify"
        )
    
    def tearDown(self):
        """Clean up after the test case."""
        os.unlink(self.temp_whitelist.name)
    
    def test_create_validator(self):
        """Test creation of a validator."""
        validator = create_validator("whitelist", self.temp_whitelist.name)
        self.assertEqual(validator.mode, "whitelist")
        self.assertIn("requests", validator.whitelist)
    
    def test_whitelist_load(self):
        """Test loading a whitelist."""
        self.assertIn("requests", self.whitelist_validator.whitelist)
        self.assertIn("pandas", self.whitelist_validator.whitelist)
        self.assertIn("numpy", self.whitelist_validator.whitelist)
    
    def test_standard_library(self):
        """Test recognition of standard library modules."""
        # Standard library validation should always pass
        for package in ["os", "sys", "math", "datetime", "collections"]:
            is_valid, _ = self.whitelist_validator.validate_package(package)
            self.assertTrue(is_valid, f"Standard library {package} should be valid in whitelist mode")
            
            is_valid, _ = self.verify_validator.validate_package(package)
            self.assertTrue(is_valid, f"Standard library {package} should be valid in verify mode")
    
    def test_whitelist_validation(self):
        """Test package validation in whitelist mode."""
        # Valid packages (in whitelist)
        for package in ["requests", "pandas", "numpy"]:
            is_valid, message = self.whitelist_validator.validate_package(package)
            self.assertTrue(is_valid, f"Package {package} should be in whitelist: {message}")
        
        # Invalid packages (not in whitelist)
        for package in ["fakelibrary", "nonexistent", "hallucinated"]:
            is_valid, message = self.whitelist_validator.validate_package(package)
            self.assertFalse(is_valid, f"Package {package} should not be in whitelist")
    
    @patch('requests.get')
    def test_verify_validation(self, mock_get):
        """Test package validation in verify mode."""
        # Setup mock responses
        valid_response = MagicMock()
        valid_response.status_code = 200
        valid_response.json.return_value = {"info": {"summary": "Test package"}}
        
        invalid_response = MagicMock()
        invalid_response.status_code = 404
        
        # Mock PyPI responses
        def mock_get_side_effect(url, *args, **kwargs):
            if "requests" in url or "numpy" in url or "pandas" in url:
                return valid_response
            elif "real-package" in url:
                return valid_response
            else:
                return invalid_response
        
        mock_get.side_effect = mock_get_side_effect
        
        # Test with packages in whitelist
        for package in ["requests", "pandas", "numpy"]:
            is_valid, message = self.verify_validator.validate_package(package)
            self.assertTrue(is_valid, f"Package {package} should be valid: {message}")
            # These should be valid from whitelist, no PyPI check needed
            mock_get.assert_not_called()
        
        mock_get.reset_mock()
        
        # Test with valid package not in whitelist
        is_valid, message = self.verify_validator.validate_package("real-package")
        self.assertTrue(is_valid, f"Package real-package should be valid: {message}")
        mock_get.assert_called_with("https://pypi.org/pypi/real-package/json", timeout=5)
        
        mock_get.reset_mock()
        
        # Test with invalid package
        is_valid, message = self.verify_validator.validate_package("fakelibrary")
        self.assertFalse(is_valid, f"Package fakelibrary should be invalid")
        mock_get.assert_called_with("https://pypi.org/pypi/fakelibrary/json", timeout=5)
    
    def test_import_validation(self):
        """Test import statement validation."""
        # Valid imports
        valid_imports = [
            "import os",
            "import sys, math",
            "import pandas as pd",
            "from numpy import array",
            "from requests import get, post"
        ]
        
        for import_stmt in valid_imports:
            is_valid, message = self.whitelist_validator.validate_import(import_stmt)
            self.assertTrue(is_valid, f"Import {import_stmt} should be valid: {message}")
        
        # Invalid imports
        invalid_imports = [
            "import fakelibrary",
            "from nonexistent import function",
            "import hallucinated as h"
        ]
        
        for import_stmt in invalid_imports:
            is_valid, message = self.whitelist_validator.validate_import(import_stmt)
            self.assertFalse(is_valid, f"Import {import_stmt} should be invalid")
    
    def test_requirements_validation(self):
        """Test requirements.txt validation."""
        # Create a requirements file with both valid and invalid packages
        requirements = """
        # Valid packages
        requests==2.28.1
        pandas>=1.3.0
        numpy
        
        # Invalid packages
        fakelibrary==1.0
        nonexistent>=2.0
        """
        
        results = self.whitelist_validator.validate_requirements(requirements)
        
        # Check results
        valid_packages = [r[0] for r in results if r[1]]
        invalid_packages = [r[0] for r in results if not r[1]]
        
        self.assertIn("requests", valid_packages)
        self.assertIn("pandas", valid_packages)
        self.assertIn("numpy", valid_packages)
        
        self.assertIn("fakelibrary", invalid_packages)
        self.assertIn("nonexistent", invalid_packages)
    
    def test_adding_to_whitelist(self):
        """Test adding packages to the whitelist."""
        # Package not initially in whitelist
        self.assertNotIn("newpackage", self.whitelist_validator.whitelist)
        
        # Add to whitelist
        metadata = {"purpose": "Test package", "pypi_url": "https://pypi.org/project/newpackage/"}
        self.whitelist_validator.add_to_whitelist("newpackage", metadata)
        
        # Should be in whitelist now
        self.assertIn("newpackage", self.whitelist_validator.whitelist)
        
        # Save whitelist to a new file
        new_whitelist_path = tempfile.mktemp(suffix='.json')
        self.whitelist_validator.save_whitelist(new_whitelist_path)
        
        # Load the saved whitelist and check it contains the new package
        with open(new_whitelist_path, 'r') as f:
            saved_whitelist = json.load(f)
        
        self.assertIn("newpackage", saved_whitelist)
        self.assertEqual(saved_whitelist["newpackage"]["purpose"], "Test package")
        
        # Clean up
        os.unlink(new_whitelist_path)
    
    def test_dependency_motivation_analysis(self):
        """Test analyzing the motivation for a dependency."""
        # Sample code with imports
        code_sample = """
        import requests
        import pandas as pd
        
        def fetch_data(url):
            response = requests.get(url)
            return pd.DataFrame(response.json())
        """
        
        # Analyze requests usage
        analysis = self.whitelist_validator.analyze_dependency_motivation(code_sample, "requests")
        
        # Check the analysis results
        self.assertEqual(analysis["package_name"], "requests")
        self.assertGreater(analysis["usage_count"], 0)
        self.assertIn("get", analysis["function_calls"])

if __name__ == '__main__':
    unittest.main()
