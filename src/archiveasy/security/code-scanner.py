"""
LLM-generated code scanner to detect potentially harmful imports.

This module scans code generated by LLMs to identify potentially harmful
or hallucianted package dependencies and prevent slopsquatting attacks.
"""
# crumb: security\code_scanner.py
import re
import logging
from typing import Dict, List, Set, Tuple, Optional, Union

from archiveasy.security.package_validator import create_validator, PackageValidator

logger = logging.getLogger(__name__)

class CodeScanner:
    """
    Scans LLM-generated code for security issues like slopsquatting.
    """
    
    def __init__(self, validator: Optional[PackageValidator] = None, 
                mode: str = "verify", whitelist_path: Optional[str] = None):
        """
        Initialize the code scanner.
        
        Args:
            validator: Optional pre-configured package validator
            mode: Validation mode if creating a new validator - "whitelist" or "verify"
            whitelist_path: Path to whitelist JSON file if creating a new validator
        """
        self.validator = validator or create_validator(mode, whitelist_path)
    
    def scan_artifact(self, artifact: Dict[str, any]) -> List[Dict[str, any]]:
        """
        Scan an artifact for security issues.
        
        Args:
            artifact: The artifact dictionary
            
        Returns:
            List of detected issues
        """
        issues = []
        
        # Only scan code artifacts
        artifact_type = artifact.get("type", "")
        if not (artifact_type in ["code", "application/vnd.ant.code"] or 
                "code" in artifact_type.lower()):
            return issues
        
        content = artifact.get("content", "")
        if not content:
            return issues
        
        # Scan for imports based on language
        language = artifact.get("language", "").lower()
        
        if language in ["python", "py"]:
            issues.extend(self._scan_python_imports(content))
        elif language in ["javascript", "js", "typescript", "ts"]:
            issues.extend(self._scan_js_imports(content))
        else:
            # Generic scan for any imports
            issues.extend(self._scan_generic_imports(content))
        
        return issues
    
    def scan_message(self, message: str) -> List[Dict[str, any]]:
        """
        Scan a message for security issues in code blocks.
        
        Args:
            message: The message text
            
        Returns:
            List of detected issues
        """
        issues = []
        
        # Find code blocks in the message
        code_blocks = re.finditer(r'```(\w*)\n(.*?)```', message, re.DOTALL)
        
        for block in code_blocks:
            language = block.group(1).lower() or "unknown"
            code = block.group(2)
            
            # Create a fake artifact for scanning
            artifact = {
                "type": "code",
                "language": language,
                "content": code
            }
            
            block_issues = self.scan_artifact(artifact)
            issues.extend(block_issues)
        
        return issues
    
    def _scan_python_imports(self, code: str) -> List[Dict[str, any]]:
        """
        Scan Python code for potentially harmful imports.
        
        Args:
            code: Python code content
            
        Returns:
            List of detected issues
        """
        issues = []
        
        # Simple import patterns
        import_patterns = [
            r'^import\s+([A-Za-z0-9_\.]+)(?:\s+as\s+[A-Za-z0-9_]+)?',
            r'^from\s+([A-Za-z0-9_\.]+)\s+import\s+'
        ]
        
        lines = code.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    full_package = match.group(1)
                    package_name = full_package.split('.')[0]
                    
                    # Validate package
                    is_valid, message = self.validator.validate_package(package_name)
                    if not is_valid:
                        issues.append({
                            "line": i + 1,
                            "code": line,
                            "package": package_name,
                            "message": message,
                            "severity": "high",
                            "type": "potentially_harmful_import"
                        })
        
        # Look for pip install commands
        pip_pattern = r'(?:pip|pip3)\s+install\s+([A-Za-z0-9_\-\.]+)'
        pip_matches = re.finditer(pip_pattern, code)
        
        for match in pip_matches:
            package_name = match.group(1)
            
            # Validate package
            is_valid, message = self.validator.validate_package(package_name)
            if not is_valid:
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    "line": line_num,
                    "code": match.group(0),
                    "package": package_name,
                    "message": message,
                    "severity": "high",
                    "type": "potentially_harmful_pip_install"
                })
        
        # Look for requirements.txt style dependencies
        req_pattern = r'^([A-Za-z0-9_\-\.]+)(?:==|>=|<=|>|<|~=)'
        for i, line in enumerate(lines):
            line = line.strip()
            match = re.match(req_pattern, line)
            if match:
                package_name = match.group(1)
                
                # Check if this looks like a requirements.txt file or section
                if i > 0 and (lines[i-1].strip() == "requirements.txt" or 
                              "requirements" in lines[i-1].lower()):
                    # Validate package
                    is_valid, message = self.validator.validate_package(package_name)
                    if not is_valid:
                        issues.append({
                            "line": i + 1,
                            "code": line,
                            "package": package_name,
                            "message": message,
                            "severity": "high",
                            "type": "potentially_harmful_requirement"
                        })
        
        return issues
    
    def _scan_js_imports(self, code: str) -> List[Dict[str, any]]:
        """
        Scan JavaScript code for potentially harmful imports.
        
        Args:
            code: JavaScript code content
            
        Returns:
            List of detected issues
        """
        issues = []
        
        # Look for npm/yarn install commands
        npm_pattern = r'(?:npm|yarn)\s+(?:add|install)\s+(?:--save\s+)?([A-Za-z0-9_\-\.]+)'
        npm_matches = re.finditer(npm_pattern, code)
        
        for match in npm_matches:
            package_name = match.group(1)
            
            # Validate package using NPM registry (currently using PyPI validation as placeholder)
            # In a full implementation, you would check against the NPM registry instead
            is_valid, message = self.validator.validate_package(package_name)
            if not is_valid:
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    "line": line_num,
                    "code": match.group(0),
                    "package": package_name,
                    "message": "Suspicious npm package - needs verification",
                    "severity": "medium",
                    "type": "potentially_harmful_npm_install"
                })
        
        # Look for package.json dependencies
        package_json_pattern = r'"([A-Za-z0-9_\-\.]+)":\s*"(?:[^"]+)"'
        for match in re.finditer(package_json_pattern, code):
            package_name = match.group(1)
            
            # Check if this looks like it's in a dependencies section
            deps_section = False
            context_start = max(0, match.start() - 50)
            context = code[context_start:match.start()]
            if "dependencies" in context or "devDependencies" in context:
                deps_section = True
            
            if deps_section:
                # Validate package (as above, using PyPI validation as placeholder)
                is_valid, message = self.validator.validate_package(package_name)
                if not is_valid:
                    line_num = code[:match.start()].count('\n') + 1
                    issues.append({
                        "line": line_num,
                        "code": match.group(0),
                        "package": package_name,
                        "message": "Suspicious npm package - needs verification",
                        "severity": "medium",
                        "type": "potentially_harmful_npm_dependency"
                    })
        
        return issues
    
    def _scan_generic_imports(self, code: str) -> List[Dict[str, any]]:
        """
        Scan generic code for potentially harmful imports.
        
        Args:
            code: Generic code content
            
        Returns:
            List of detected issues
        """
        issues = []
        
        # Look for common package manager commands
        package_manager_patterns = [
            (r'pip\s+install\s+([A-Za-z0-9_\-\.]+)', "pip"),
            (r'npm\s+(?:install|add)\s+([A-Za-z0-9_\-\.]+)', "npm"),
            (r'yarn\s+add\s+([A-Za-z0-9_\-\.]+)', "yarn"),
            (r'gem\s+install\s+([A-Za-z0-9_\-\.]+)', "gem"),
            (r'composer\s+require\s+([A-Za-z0-9_\-\.\/]+)', "composer"),
            (r'go\s+get\s+([A-Za-z0-9_\-\.\/]+)', "go")
        ]
        
        for pattern, manager in package_manager_patterns:
            for match in re.finditer(pattern, code):
                package_name = match.group(1)
                line_num = code[:match.start()].count('\n') + 1
                
                # For simplicity, we'll use the PyPI validation for all package types
                # In a real implementation, you'd check against the appropriate registry
                is_valid, message = self.validator.validate_package(package_name)
                
                if not is_valid:
                    issues.append({
                        "line": line_num,
                        "code": match.group(0),
                        "package": package_name,
                        "message": f"Suspicious {manager} package - needs verification",
                        "severity": "medium",
                        "type": f"potentially_harmful_{manager}_install"
                    })
        
        return issues

# Utility functions
def scan_artifacts(artifacts: List[Dict[str, any]], 
                  mode: str = "verify", 
                  whitelist_path: Optional[str] = None) -> List[Dict[str, any]]:
    """
    Scan artifacts for security issues.
    
    Args:
        artifacts: List of artifacts
        mode: Validation mode - "whitelist" or "verify"
        whitelist_path: Path to whitelist JSON file (optional)
        
    Returns:
        List of detected issues
    """
    scanner = CodeScanner(mode=mode, whitelist_path=whitelist_path)
    
    all_issues = []
    for artifact in artifacts:
        issues = scanner.scan_artifact(artifact)
        all_issues.extend(issues)
    
    return all_issues

def scan_message(message: str,
                mode: str = "verify", 
                whitelist_path: Optional[str] = None) -> List[Dict[str, any]]:
    """
    Scan a message for security issues in code blocks.
    
    Args:
        message: Message text
        mode: Validation mode - "whitelist" or "verify"
        whitelist_path: Path to whitelist JSON file (optional)
        
    Returns:
        List of detected issues
    """
    scanner = CodeScanner(mode=mode, whitelist_path=whitelist_path)
    return scanner.scan_message(message)