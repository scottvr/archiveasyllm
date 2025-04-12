"""
Package validation module to protect against slopsquatting (typosquatting targeted at LLM-generated code).

This module provides two levels of package validation:
1. Strict whitelist mode: Only allows pre-approved packages
2. Verification mode: Validates packages against PyPI and analyzes their intended purpose
"""
import os
import json
import re
import logging
from datetime import datetime
import requests
from typing import Dict, List, Set, Tuple, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class PackageValidator:
    """
    Validates Python package imports and requirements to protect against slopsquatting.
    """
    
    def __init__(self, whitelist_path: Optional[str] = None, mode: str = "verify"):
        """
        Initialize the package validator.
        
        Args:
            whitelist_path: Path to whitelist JSON file (optional)
            mode: Validation mode - "whitelist" or "verify" (default: "verify")
        """
        self.mode = mode
        
        # Load whitelist
        self.whitelist = set()
        self.package_metadata = {}
        
        if whitelist_path:
            self._load_whitelist(whitelist_path)
        else:
            # Default whitelist path
            default_path = Path(__file__).parent / "conf" / "package_whitelist.json"
            if default_path.exists():
                self._load_whitelist(str(default_path))
            else:
                # Initialize with common safe packages
                self._initialize_default_whitelist()
    
    def _load_whitelist(self, whitelist_path: str) -> None:
        """
        Load package whitelist from a JSON file.
        
        Args:
            whitelist_path: Path to whitelist JSON file
        """
        try:
            with open(whitelist_path, 'r') as f:
                whitelist_data = json.load(f)
                
            # Extract package names for quick lookup
            self.whitelist = set(whitelist_data.keys())
            
            # Store full metadata
            self.package_metadata = whitelist_data
            
            logger.info(f"Loaded {len(self.whitelist)} packages in whitelist")
            
        except Exception as e:
            logger.error(f"Error loading whitelist: {e}")
            # Initialize defaults
            self._initialize_default_whitelist()
    
    def _initialize_default_whitelist(self) -> None:
        """Initialize with a default set of common Python packages."""
        default_packages = {
            # Standard libraries
            "os": {"purpose": "Operating system interfaces", "standard_lib": True},
            "sys": {"purpose": "System-specific parameters and functions", "standard_lib": True},
            "math": {"purpose": "Mathematical functions", "standard_lib": True},
            "json": {"purpose": "JSON encoding and decoding", "standard_lib": True},
            "datetime": {"purpose": "Date and time handling", "standard_lib": True},
            "collections": {"purpose": "Specialized container datatypes", "standard_lib": True},
            "re": {"purpose": "Regular expression operations", "standard_lib": True},
            "pathlib": {"purpose": "Object-oriented filesystem paths", "standard_lib": True},
            "typing": {"purpose": "Support for type hints", "standard_lib": True},
            "uuid": {"purpose": "UUID objects", "standard_lib": True},
            "logging": {"purpose": "Logging facility", "standard_lib": True},
            "threading": {"purpose": "Thread-based parallelism", "standard_lib": True},
            "multiprocessing": {"purpose": "Process-based parallelism", "standard_lib": True},
            "time": {"purpose": "Time access and conversions", "standard_lib": True},
            "random": {"purpose": "Generate pseudo-random numbers", "standard_lib": True},
            "argparse": {"purpose": "Command-line option and argument parsing", "standard_lib": True},
            "csv": {"purpose": "CSV file reading and writing", "standard_lib": True},
            "io": {"purpose": "Core tools for working with streams", "standard_lib": True},
            "shutil": {"purpose": "High-level file operations", "standard_lib": True},
            "pickle": {"purpose": "Python object serialization", "standard_lib": True},
            "hashlib": {"purpose": "Secure hash and message digest algorithms", "standard_lib": True},
            "itertools": {"purpose": "Functions creating iterators for efficient looping", "standard_lib": True},
            "functools": {"purpose": "Higher-order functions and operations", "standard_lib": True},
            "contextlib": {"purpose": "Utilities for with-statement contexts", "standard_lib": True},
            "unittest": {"purpose": "Unit testing framework", "standard_lib": True},
            
            # Common third-party libraries
            "flask": {"purpose": "Web microframework", "pypi_url": "https://pypi.org/project/Flask/"},
            "requests": {"purpose": "HTTP library", "pypi_url": "https://pypi.org/project/requests/"},
            "numpy": {"purpose": "Scientific computing", "pypi_url": "https://pypi.org/project/numpy/"},
            "pandas": {"purpose": "Data analysis and manipulation", "pypi_url": "https://pypi.org/project/pandas/"},
            "matplotlib": {"purpose": "Visualization library", "pypi_url": "https://pypi.org/project/matplotlib/"},
            "scipy": {"purpose": "Scientific computing", "pypi_url": "https://pypi.org/project/scipy/"},
            "pytest": {"purpose": "Testing framework", "pypi_url": "https://pypi.org/project/pytest/"},
            "django": {"purpose": "Web framework", "pypi_url": "https://pypi.org/project/Django/"},
            "sqlalchemy": {"purpose": "SQL toolkit and ORM", "pypi_url": "https://pypi.org/project/SQLAlchemy/"},
            "tensorflow": {"purpose": "Machine learning framework", "pypi_url": "https://pypi.org/project/tensorflow/"},
            "pytorch": {"purpose": "Machine learning framework", "pypi_url": "https://pypi.org/project/torch/"},
            "scikit-learn": {"purpose": "Machine learning library", "pypi_url": "https://pypi.org/project/scikit-learn/"},
            "transformers": {"purpose": "Natural language processing", "pypi_url": "https://pypi.org/project/transformers/"},
            "pillow": {"purpose": "Image processing", "pypi_url": "https://pypi.org/project/Pillow/"},
            "beautifulsoup4": {"purpose": "HTML/XML parsing", "pypi_url": "https://pypi.org/project/beautifulsoup4/"},
            "neo4j": {"purpose": "Neo4j graph database driver", "pypi_url": "https://pypi.org/project/neo4j/"},
            "faiss-cpu": {"purpose": "Efficient similarity search", "pypi_url": "https://pypi.org/project/faiss-cpu/"},
            "sentence-transformers": {"purpose": "Sentence embeddings", "pypi_url": "https://pypi.org/project/sentence-transformers/"},
            "python-dotenv": {"purpose": "Environment variable handling", "pypi_url": "https://pypi.org/project/python-dotenv/"},
            "pydantic": {"purpose": "Data validation", "pypi_url": "https://pypi.org/project/pydantic/"},
            "gunicorn": {"purpose": "WSGI HTTP server", "pypi_url": "https://pypi.org/project/gunicorn/"},
            "flask-restx": {"purpose": "Flask REST API framework", "pypi_url": "https://pypi.org/project/flask-restx/"},
            "anthropic": {"purpose": "Anthropic Claude API client", "pypi_url": "https://pypi.org/project/anthropic/"},
            "openai": {"purpose": "OpenAI API client", "pypi_url": "https://pypi.org/project/openai/"},
            "rich": {"purpose": "Rich text formatting", "pypi_url": "https://pypi.org/project/rich/"}
        }
        
        self.whitelist = set(default_packages.keys())
        self.package_metadata = default_packages
        
        logger.info(f"Initialized default whitelist with {len(self.whitelist)} packages")
    
    def save_whitelist(self, path: str) -> bool:
        """
        Save the current whitelist to a file.
        
        Args:
            path: Path to save the whitelist JSON file
            
        Returns:
            True if saved successfully
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(self.package_metadata, f, indent=2, sort_keys=True)
                
            logger.info(f"Saved whitelist with {len(self.whitelist)} packages to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving whitelist: {e}")
            return False
    
    def add_to_whitelist(self, package_name: str, metadata: Dict[str, any]) -> bool:
        """
        Add a package to the whitelist.
        
        Args:
            package_name: Package name
            metadata: Package metadata (purpose, pypi_url, etc.)
            
        Returns:
            True if added successfully
        """
        self.whitelist.add(package_name)
        self.package_metadata[package_name] = metadata
        return True
    
    def validate_import(self, import_statement: str) -> Tuple[bool, str]:
        """
        Validate an import statement.
        
        Args:
            import_statement: Python import statement
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Extract package name from import statement
        package_name = self._extract_package_name(import_statement)
        
        if not package_name:
            return False, f"Could not extract package name from import: {import_statement}"
        
        return self.validate_package(package_name)
    
    def validate_requirements(self, requirements_file: str) -> List[Tuple[str, bool, str]]:
        """
        Validate a requirements.txt file.
        
        Args:
            requirements_file: Content of requirements.txt file
            
        Returns:
            List of (package_name, is_valid, message) tuples
        """
        results = []
        
        # Parse requirements file
        lines = requirements_file.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Extract package name (ignore version constraints)
            package_match = re.match(r'^([A-Za-z0-9_\-\.]+)', line)
            if package_match:
                package_name = package_match.group(1)
                is_valid, message = self.validate_package(package_name)
                results.append((package_name, is_valid, message))
            else:
                results.append((line, False, f"Could not parse requirement: {line}"))
        
        return results
    
    def validate_package(self, package_name: str) -> Tuple[bool, str]:
        """
        Validate a package name.
        
        Args:
            package_name: Package name
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Clean package name (remove submodules)
        base_package = package_name.split('.')[0]
        
        # Whitelist mode
        if self.mode == "whitelist":
            if base_package in self.whitelist:
                return True, f"Package '{base_package}' is in the whitelist"
            else:
                return False, f"Package '{base_package}' is not in the whitelist"
        
        # Verify mode
        elif self.mode == "verify":
            # Check whitelist first
            if base_package in self.whitelist:
                return True, f"Package '{base_package}' is in the whitelist"
            
            # Verify against PyPI
            is_valid, message, metadata = self._verify_package_exists(base_package)
            
            # Add to whitelist if valid
            if is_valid and metadata:
                self.add_to_whitelist(base_package, metadata)
                
            return is_valid, message
        
        # Invalid mode
        else:
            return False, f"Invalid validation mode: {self.mode}"
    
    def _extract_package_name(self, import_statement: str) -> Optional[str]:
        """
        Extract package name from an import statement.
        
        Args:
            import_statement: Python import statement
            
        Returns:
            Package name or None if extraction failed
        """
        # Match various import patterns
        patterns = [
            r'^import\s+([A-Za-z0-9_\.]+)',  # import package
            r'^from\s+([A-Za-z0-9_\.]+)\s+import',  # from package import ...
            r'^import\s+([A-Za-z0-9_\.]+)\s+as',  # import package as ...
        ]
        
        for pattern in patterns:
            match = re.match(pattern, import_statement.strip())
            if match:
                # Get base package (first part of dotted path)
                return match.group(1).split('.')[0]
        
        return None
    
    def _verify_package_exists(self, package_name: str) -> Tuple[bool, str, Optional[Dict[str, any]]]:
        """
        Verify that a package exists on PyPI and gather metadata.
        
        Args:
            package_name: Package name
            
        Returns:
            Tuple of (is_valid, message, metadata)
        """
        # Skip standard library packages
        if self._is_standard_library(package_name):
            return True, f"Package '{package_name}' is a Python standard library", {
                "purpose": "Python standard library",
                "standard_lib": True
            }
        
        # Check PyPI
        try:
            response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract metadata
                metadata = {
                    "purpose": data.get("info", {}).get("summary", ""),
                    "pypi_url": f"https://pypi.org/project/{package_name}/",
                    "verified_at": datetime.now().isoformat()
                }
                
                return True, f"Package '{package_name}' exists on PyPI", metadata
            else:
                return False, f"Package '{package_name}' does not exist on PyPI", None
                
        except Exception as e:
            logger.warning(f"Error verifying package '{package_name}': {e}")
            return False, f"Error verifying package '{package_name}': {str(e)}", None
    
    def _is_standard_library(self, package_name: str) -> bool:
        """
        Check if a package is part of the Python standard library.
        
        Args:
            package_name: Package name
            
        Returns:
            True if the package is part of the standard library
        """
        # List of standard library modules
        standard_lib = {
            "abc", "aifc", "argparse", "array", "ast", "asyncio", "base64", "bdb", "binascii",
            "bisect", "builtins", "bz2", "cProfile", "calendar", "cgi", "cgitb", "chunk", "cmath",
            "cmd", "code", "codecs", "codeop", "collections", "colorsys", "compileall", "concurrent",
            "configparser", "contextlib", "copy", "copyreg", "crypt", "csv", "ctypes", "curses",
            "dataclasses", "datetime", "dbm", "decimal", "difflib", "dis", "distutils", "doctest",
            "email", "encodings", "ensurepip", "enum", "errno", "faulthandler", "fcntl", "filecmp",
            "fileinput", "fnmatch", "formatter", "fractions", "ftplib", "functools", "gc", "getopt",
            "getpass", "gettext", "glob", "grp", "gzip", "hashlib", "heapq", "hmac", "html",
            "http", "idlelib", "imaplib", "imghdr", "imp", "importlib", "inspect", "io", "ipaddress",
            "itertools", "json", "keyword", "lib2to3", "linecache", "locale", "logging", "lzma",
            "macpath", "mailbox", "mailcap", "marshal", "math", "mimetypes", "mmap", "modulefinder",
            "msilib", "msvcrt", "multiprocessing", "netrc", "nis", "nntplib", "numbers", "operator",
            "optparse", "os", "parser", "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil",
            "platform", "plistlib", "poplib", "posix", "pprint", "profile", "pstats", "pty", "pwd",
            "py_compile", "pyclbr", "pydoc", "queue", "quopri", "random", "re", "readline", "reprlib",
            "resource", "rlcompleter", "runpy", "sched", "secrets", "select", "selectors", "shelve",
            "shlex", "shutil", "signal", "site", "smtpd", "smtplib", "sndhdr", "socket", "socketserver",
            "spwd", "sqlite3", "ssl", "stat", "statistics", "string", "stringprep", "struct", "subprocess",
            "sunau", "symbol", "symtable", "sys", "sysconfig", "syslog", "tabnanny", "tarfile", "telnetlib",
            "tempfile", "termios", "test", "textwrap", "threading", "time", "timeit", "tkinter", "token",
            "tokenize", "trace", "traceback", "tracemalloc", "tty", "turtle", "turtledemo", "types",
            "typing", "unicodedata", "unittest", "urllib", "uu", "uuid", "venv", "warnings", "wave",
            "weakref", "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc",
            "zipapp", "zipfile", "zipimport", "zlib"
        }
        
        return package_name in standard_lib
    
    def analyze_dependency_motivation(self, code_sample: str, package_name: str) -> Dict[str, any]:
        """
        Analyze the motivation for including a dependency in code.
        
        Args:
            code_sample: Sample of code using the package
            package_name: Package name
            
        Returns:
            Dictionary with analysis results
        """
        # This would ideally use an LLM to analyze the motivation, but here's a simple version
        
        # Look for import statement
        import_pattern = re.compile(f"(?:import\\s+{package_name}|from\\s+{package_name}\\s+import)", re.MULTILINE)
        import_matches = import_pattern.findall(code_sample)
        
        # Look for package usage
        usage_pattern = re.compile(f"{package_name}\\.[A-Za-z0-9_]+", re.MULTILINE)
        usage_matches = usage_pattern.findall(code_sample)
        
        # Extract function/method calls
        function_calls = set()
        for usage in usage_matches:
            parts = usage.split('.')
            if len(parts) > 1:
                function_calls.add(parts[1])
        
        # Basic analysis
        analysis = {
            "package_name": package_name,
            "import_statements": import_matches,
            "usage_count": len(usage_matches),
            "function_calls": list(function_calls),
            "motivation": "Unknown"
        }
        
        # Determine motivation based on usage patterns
        if package_name in self.package_metadata:
            analysis["motivation"] = self.package_metadata[package_name].get("purpose", "Unknown")
        elif function_calls:
            analysis["motivation"] = f"Used for: {', '.join(function_calls)}"
        
        return analysis

# Function to easily create a validator instance
def create_validator(mode: str = "verify", whitelist_path: Optional[str] = None) -> PackageValidator:
    """
    Create a package validator instance.
    
    Args:
        mode: Validation mode - "whitelist" or "verify"
        whitelist_path: Path to whitelist JSON file (optional)
        
    Returns:
        PackageValidator instance
    """
    return PackageValidator(whitelist_path=whitelist_path, mode=mode)
