"""
Requirements analyzer for validating project dependencies.

This module scans requirements files and Python code to identify imported packages
and validate them against the package validator to prevent slopsquatting.
"""
# crumb: security\analyzer.py
import os
import re
import ast
import logging
from typing import Dict, List, Set, Tuple, Optional, Union
from pathlib import Path
from collections import defaultdict

from archiveasy.security.validator import PackageValidator

logger = logging.getLogger(__name__)

class RequirementsAnalyzer:
    """
    Analyzes project dependencies for security issues like slopsquatting.
    """
    
    def __init__(self, validator: PackageValidator):
        """
        Initialize the requirements analyzer.
        
        Args:
            validator: Package validator instance
        """
        self.validator = validator
        self.found_imports = set()
        self.found_requirements = set()
        self.invalid_packages = []
        self.package_usage = defaultdict(list)
    
    def analyze_project(self, project_path: str) -> Dict[str, any]:
        """
        Analyze a project's dependencies.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            Dictionary with analysis results
        """
        # Reset state
        self.found_imports = set()
        self.found_requirements = set()
        self.invalid_packages = []
        self.package_usage = defaultdict(list)
        
        # Find and analyze requirements files
        requirements_files = self._find_requirements_files(project_path)
        for req_file in requirements_files:
            self._analyze_requirements_file(req_file)
        
        # Find and analyze Python files
        python_files = self._find_python_files(project_path)
        for py_file in python_files:
            self._analyze_python_file(py_file)
        
        # Analyze package motivations
        package_motivations = {}
        for package, usages in self.package_usage.items():
            if usages:
                # Take the first usage as a sample
                sample_usage = usages[0]["code"]
                # Analyze motivation
                motivation = self.validator.analyze_dependency_motivation(sample_usage, package)
                package_motivations[package] = motivation
        
        return {
            "requirements_files": len(requirements_files),
            "python_files": len(python_files),
            "packages": {
                "imports": list(self.found_imports),
                "requirements": list(self.found_requirements),
                "invalid": self.invalid_packages,
                "motivations": package_motivations
            }
        }
    
    def _find_requirements_files(self, project_path: str) -> List[str]:
        """
        Find requirements files in a project.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            List of paths to requirements files
        """
        requirements_files = []
        
        for root, dirs, files in os.walk(project_path):
            # Skip directories that are typically not part of the project source
            if any(excluded in root for excluded in ['/venv/', '/.venv/', '/env/', '/__pycache__/', '/.git/']):
                continue
                
            for file in files:
                if file == 'requirements.txt' or file.endswith('-requirements.txt'):
                    requirements_files.append(os.path.join(root, file))
        
        return requirements_files
    
    def _find_python_files(self, project_path: str) -> List[str]:
        """
        Find Python files in a project.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            List of paths to Python files
        """
        python_files = []
        
        for root, dirs, files in os.walk(project_path):
            # Skip directories that are typically not part of the project source
            if any(excluded in root for excluded in ['/venv/', '/.venv/', '/env/', '/__pycache__/', '/.git/']):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return python_files
    
    def _analyze_requirements_file(self, file_path: str) -> None:
        """
        Analyze a requirements file.
        
        Args:
            file_path: Path to the requirements file
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Validate requirements
            validation_results = self.validator.validate_requirements(content)
            
            for package_name, is_valid, message in validation_results:
                self.found_requirements.add(package_name)
                
                if not is_valid:
                    self.invalid_packages.append({
                        "name": package_name,
                        "source": file_path,
                        "error": message,
                        "type": "requirement"
                    })
            
        except Exception as e:
            logger.error(f"Error analyzing requirements file {file_path}: {e}")
    
    def _analyze_python_file(self, file_path: str) -> None:
        """
        Analyze imports in a Python file.
        
        Args:
            file_path: Path to the Python file
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract imports using ast if possible
            try:
                imports = self._extract_imports_with_ast(content)
            except SyntaxError:
                # Fall back to regex-based extraction
                imports = self._extract_imports_with_regex(content)
            
            # Validate imports
            for import_statement, package_name in imports:
                self.found_imports.add(package_name)
                
                # Record package usage
                self.package_usage[package_name].append({
                    "file": file_path,
                    "code": content
                })
                
                # Validate package
                is_valid, message = self.validator.validate_package(package_name)
                if not is_valid:
                    self.invalid_packages.append({
                        "name": package_name,
                        "source": file_path,
                        "error": message,
                        "type": "import"
                    })
            
        except Exception as e:
            logger.error(f"Error analyzing Python file {file_path}: {e}")
    
    def _extract_imports_with_ast(self, content: str) -> List[Tuple[str, str]]:
        """
        Extract imports from Python code using AST.
        
        Args:
            content: Python code content
            
        Returns:
            List of (import_statement, package_name) tuples
        """
        imports = []
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    package_name = name.name.split('.')[0]
                    import_statement = f"import {name.name}"
                    imports.append((import_statement, package_name))
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    package_name = node.module.split('.')[0]
                    names_str = ", ".join(n.name for n in node.names)
                    import_statement = f"from {node.module} import {names_str}"
                    imports.append((import_statement, package_name))
        
        return imports
    
    def _extract_imports_with_regex(self, content: str) -> List[Tuple[str, str]]:
        """
        Extract imports from Python code using regex.
        
        Args:
            content: Python code content
            
        Returns:
            List of (import_statement, package_name) tuples
        """
        imports = []
        
        # Find import statements
        import_patterns = [
            r'^import\s+([A-Za-z0-9_\.]+)(?:\s+as\s+[A-Za-z0-9_]+)?',
            r'^from\s+([A-Za-z0-9_\.]+)\s+import\s+'
        ]
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    full_package = match.group(1)
                    package_name = full_package.split('.')[0]
                    imports.append((line, package_name))
        
        return imports
    
    def generate_report(self) -> str:
        """
        Generate a report of the analysis.
        
        Returns:
            Report text
        """
        all_packages = sorted(list(self.found_imports.union(self.found_requirements)))
        invalid_packages = sorted([p["name"] for p in self.invalid_packages])
        
        report = []
        report.append("# Dependency Analysis Report")
        report.append("")
        report.append(f"Found {len(all_packages)} unique packages:")
        report.append("")
        
        # List all packages
        for package in all_packages:
            status = "❌ INVALID" if package in invalid_packages else "✅ Valid"
            report.append(f"- {package}: {status}")
        
        # Detail invalid packages
        if invalid_packages:
            report.append("")
            report.append("## Invalid Packages")
            report.append("")
            
            for invalid in self.invalid_packages:
                report.append(f"### {invalid['name']} ({invalid['type']})")
                report.append(f"- Source: {invalid['source']}")
                report.append(f"- Error: {invalid['error']}")
                report.append("")
        
        # Package motivations
        report.append("## Package Motivations")
        report.append("")
        
        for package in all_packages:
            if package in self.package_usage:
                motivation = self.validator.analyze_dependency_motivation("", package)
                purpose = motivation.get("motivation", "Unknown")
                usages = len(self.package_usage[package])
                report.append(f"- {package}: {purpose} (used in {usages} files)")
        
        return "\n".join(report)


def analyze_project_dependencies(project_path: str, 
                                mode: str = "verify", 
                                whitelist_path: Optional[str] = None,
                                save_report: bool = False,
                                report_path: Optional[str] = None) -> Dict[str, any]:
    """
    Analyze a project's dependencies for security issues.
    
    Args:
        project_path: Path to the project directory
        mode: Validation mode - "whitelist" or "verify"
        whitelist_path: Path to whitelist JSON file (optional)
        save_report: Whether to save a report file
        report_path: Path to save the report (optional)
        
    Returns:
        Analysis results
    """
    # Create validator and analyzer
    validator = PackageValidator(whitelist_path=whitelist_path, mode=mode)
    analyzer = RequirementsAnalyzer(validator)
    
    # Analyze project
    results = analyzer.analyze_project(project_path)
    
    # Generate and save report if requested
    if save_report:
        report = analyzer.generate_report()
        
        if report_path:
            report_file = report_path
        else:
            report_file = os.path.join(project_path, "dependency_report.md")
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Saved dependency report to {report_file}")
    
    return results