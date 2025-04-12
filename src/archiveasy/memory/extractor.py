"""
Knowledge extractor module for identifying architectural patterns and decisions in LLM responses.
"""
# crumb: memory\extractor.py
from typing import Dict, Any, List, Optional
import re
import uuid
import json
import ast
import logging

logger = logging.getLogger(__name__)

class KnowledgeExtractor:
    """
    Extracts knowledge elements (entities, relationships, decisions, patterns)
    from LLM responses and artifacts.
    """
    
    def extract(self, response: str, artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract knowledge from an LLM response and its artifacts.
        
        Args:
            response: The text response from the LLM
            artifacts: List of artifacts (code, diagrams, etc.)
            
        Returns:
            Dictionary with extracted knowledge elements
        """
        # Initialize result structure
        result = {
            "entities": [],
            "relationships": [],
            "decisions": [],
            "patterns": []
        }
        
        # Extract from text response
        self._extract_from_text(response, result)
        
        # Extract from artifacts
        for artifact in artifacts:
            self._extract_from_artifact(artifact, result)
        
        # Process the extracted knowledge
        self.deduplicate_knowledge(result)
        self.resolve_relationships(result)
        
        return result
    
    def _extract_from_text(self, text: str, result: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract knowledge elements from text response.
        
        Args:
            text: The LLM response text
            result: Dictionary to populate with extracted elements
        """
        # Extract architectural decisions
        decisions = self._extract_decisions(text)
        result["decisions"].extend(decisions)
        
        # Extract design patterns
        patterns = self._extract_patterns(text)
        result["patterns"].extend(patterns)
        
        # Basic entity extraction (functions, classes, variables)
        entities = self._extract_entities(text)
        result["entities"].extend(entities)
        
        # Extract relationships between entities
        relationships = self._extract_relationships(text)
        result["relationships"].extend(relationships)
    
    def _extract_from_artifact(self, artifact: Dict[str, Any], result: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract knowledge elements from an artifact.
        
        Args:
            artifact: The artifact dictionary
            result: Dictionary to populate with extracted elements
        """
        artifact_type = artifact.get("type", "")
        content = artifact.get("content", "")
        
        if not content:
            return
        
        # Handle code artifacts
        if artifact_type in ["code", "application/vnd.ant.code"]:
            # Extract code entities and relationships
            language = artifact.get("language", "")
            
            if language.lower() in ["python", "py"]:
                self._extract_from_python_code(content, result)
            elif language.lower() in ["javascript", "js", "typescript", "ts"]:
                self._extract_from_js_code(content, result)
            else:
                # Generic code extraction
                self._extract_from_generic_code(content, result)
        
        # Handle markdown artifacts
        elif artifact_type in ["markdown", "text/markdown"]:
            self._extract_from_markdown(content, result)
    
    def _extract_decisions(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract architectural decisions from text.
        
        Looks for patterns like:
        - "I decided to use X because Y"
        - "We should use X for Y reason"
        - "The architecture will use X approach for Y component"
        - "X was chosen over Y because Z"
        
        Args:
            text: The text to analyze
            
        Returns:
            List of extracted decision dictionaries
        """
        decisions = []
        
        # Pattern matchers for decision statements
        decision_patterns = [
            r"(?:I|we) (?:decided|chose|selected|opted|recommend) to use ([^,\.]+) (?:because|since|as) ([^\.]+)",
            r"(?:I|we) (?:will|should|could|might) use ([^,\.]+) for ([^\.]+)",
            r"(?:The|This) (?:architecture|system|framework|approach) (?:will|should|could|might) use ([^,\.]+) (?:for|to) ([^\.]+)",
            r"([A-Za-z0-9_\-\s]+) (?:was|is) chosen over ([^,\.]+) because ([^\.]+)",
            r"(?:I|we) (?:recommend|suggest|propose) ([^,\.]+) (?:for|to|because|since) ([^\.]+)"
        ]
        
        for pattern in decision_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract components based on the pattern
                if len(match.groups()) == 2:
                    choice, reasoning = match.groups()
                    alternatives = []
                elif len(match.groups()) == 3:
                    choice, alternatives, reasoning = match.groups()
                    alternatives = [alt.strip() for alt in alternatives.split(",")]
                else:
                    continue
                
                decision = {
                    "id": str(uuid.uuid4()),
                    "title": f"Decision to use {choice.strip()}",
                    "description": match.group(0),
                    "reasoning": reasoning.strip(),
                    "alternatives": alternatives if isinstance(alternatives, list) else [],
                    "affects": []  # To be filled later if entities are linked
                }
                
                decisions.append(decision)
        
        return decisions
    
    def _extract_patterns(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract design patterns from text.
        
        Looks for patterns like:
        - "Follow the X pattern for Y"
        - "Using the X pattern to solve Y"
        - "Implement X using the Y pattern"
        
        Args:
            text: The text to analyze
            
        Returns:
            List of extracted pattern dictionaries
        """
        patterns = []
        
        # Pattern matchers for design pattern statements
        pattern_matchers = [
            r"(?:follow|use|implement|apply) the ([A-Za-z0-9_\-\s]+) pattern for ([^\.]+)",
            r"using the ([A-Za-z0-9_\-\s]+) pattern to ([^\.]+)",
            r"implement ([^,\.]+) using the ([A-Za-z0-9_\-\s]+) pattern",
            r"([A-Za-z0-9_\-\s]+) pattern (?:is|will be|should be) used for ([^\.]+)",
            r"adopt(?:ing)? (?:a|the) ([A-Za-z0-9_\-\s]+) pattern for ([^\.]+)"
        ]
        
        for matcher in pattern_matchers:
            matches = re.finditer(matcher, text, re.IGNORECASE)
            for match in matches:
                # Extract components based on the pattern
                if len(match.groups()) == 2:
                    pattern_name, description = match.groups()
                    
                    # The pattern might be reversed in some regex
                    if "using the" in matcher and "pattern" not in pattern_name:
                        pattern_name, description = description, pattern_name
                    
                    pattern_dict = {
                        "id": str(uuid.uuid4()),
                        "name": pattern_name.strip(),
                        "description": f"{match.group(0)}",
                        "examples": []  # To be filled with code examples if found
                    }
                    
                    patterns.append(pattern_dict)
        
        return patterns
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract code entities from text.
        
        Looks for mentions of:
        - Functions/methods
        - Classes
        - Variables
        - Modules
        
        Args:
            text: The text to analyze
            
        Returns:
            List of extracted entity dictionaries
        """
        entities = []
        
        # Entity patterns
        entity_patterns = {
            "function": r"(?:function|method|def) `?([A-Za-z0-9_]+)`?",
            "class": r"(?:class) `?([A-Za-z0-9_]+)`?",
            "variable": r"(?:variable|var|let|const) `?([A-Za-z0-9_]+)`?",
            "module": r"(?:module|import) `?([A-Za-z0-9_\.]+)`?"
        }
        
        for entity_type, pattern in entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_name = match.group(1)
                
                entity = {
                    "id": str(uuid.uuid4()),
                    "name": entity_name,
                    "type": entity_type,
                    "properties": {
                        "mentioned_in_text": True,
                        "context": text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
                    }
                }
                
                entities.append(entity)
        
        return entities
    
    def _extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities from text.
        
        Looks for patterns like:
        - "X depends on Y"
        - "X calls Y"
        - "X inherits from Y"
        - "X uses Y"
        
        Args:
            text: The text to analyze
            
        Returns:
            List of extracted relationship dictionaries
        """
        relationships = []
        
        # Relationship patterns
        relationship_patterns = {
            "DEPENDS_ON": r"([A-Za-z0-9_]+) (?:depends on|requires|needs) ([A-Za-z0-9_]+)",
            "CALLS": r"([A-Za-z0-9_]+) (?:calls|invokes|executes) ([A-Za-z0-9_]+)",
            "INHERITS_FROM": r"([A-Za-z0-9_]+) (?:inherits from|extends|subclasses) ([A-Za-z0-9_]+)",
            "USES": r"([A-Za-z0-9_]+) (?:uses|utilizes|leverages) ([A-Za-z0-9_]+)",
            "IMPLEMENTS": r"([A-Za-z0-9_]+) (?:implements|realizes) ([A-Za-z0-9_]+)",
            "CONTAINS": r"([A-Za-z0-9_]+) (?:contains|includes|has) ([A-Za-z0-9_]+)"
        }
        
        for rel_type, pattern in relationship_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                from_entity, to_entity = match.groups()
                
                relationship = {
                    "from_id": None,  # Will be resolved later when entities are fully processed
                    "to_id": None,    # Will be resolved later when entities are fully processed
                    "from_name": from_entity.strip(),
                    "to_name": to_entity.strip(),
                    "type": rel_type,
                    "properties": {
                        "context": text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
                    }
                }
                
                relationships.append(relationship)
        
        return relationships
    
    def _extract_from_python_code(self, code: str, result: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract knowledge elements from Python code.
        
        Args:
            code: Python code content
            result: Dictionary to populate with extracted elements
        """
        # Extract classes
        class_pattern = r"class\s+([A-Za-z0-9_]+)(?:\s*\(([A-Za-z0-9_,\s]+)\))?"
        class_matches = re.finditer(class_pattern, code)
        
        for match in class_matches:
            class_name = match.group(1)
            
            # Check for inheritance
            if match.group(2):
                parent_classes = [parent.strip() for parent in match.group(2).split(',')]
            else:
                parent_classes = []
            
            # Add class entity
            class_entity = {
                "id": str(uuid.uuid4()),
                "name": class_name,
                "type": "class",
                "properties": {
                    "language": "python",
                    "parent_classes": parent_classes
                }
            }
            
            result["entities"].append(class_entity)
            
            # Add inheritance relationships
            for parent_class in parent_classes:
                if parent_class and parent_class not in ["object", "Object"]:
                    relationship = {
                        "from_id": None,  # Will be resolved later
                        "to_id": None,    # Will be resolved later
                        "from_name": class_name,
                        "to_name": parent_class,
                        "type": "INHERITS_FROM",
                        "properties": {}
                    }
                    
                    result["relationships"].append(relationship)
        
        # Extract functions and methods
        func_pattern = r"def\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)(?:\s*->\s*([A-Za-z0-9_\[\],\s]+))?"
        func_matches = re.finditer(func_pattern, code)
        
        for match in func_matches:
            func_name = match.group(1)
            params = match.group(2).strip() if match.group(2) else ""
            return_type = match.group(3).strip() if match.group(3) else "None"
            
            # Determine if it's a method or standalone function
            # Check for indentation before the def
            line_start = code[:match.start()].rfind('\n') + 1
            indentation = match.start() - line_start
            
            func_type = "method" if indentation > 0 else "function"
            
            # Check if it's a special method
            is_special = func_name.startswith('__') and func_name.endswith('__')
            
            # Add function/method entity
            func_entity = {
                "id": str(uuid.uuid4()),
                "name": func_name,
                "type": func_type,
                "properties": {
                    "language": "python",
                    "parameters": params,
                    "return_type": return_type,
                    "is_special_method": is_special
                }
            }
            
            result["entities"].append(func_entity)
            
        # Extract imports for module dependencies
        import_pattern = r"(?:from\s+([A-Za-z0-9_.]+)\s+)?import\s+([A-Za-z0-9_,\s]+)(?:\s+as\s+([A-Za-z0-9_]+))?"
        import_matches = re.finditer(import_pattern, code)
        
        for match in import_matches:
            from_module = match.group(1) if match.group(1) else ""
            imports = [imp.strip() for imp in match.group(2).split(',')]
            alias = match.group(3) if match.group(3) else ""
            
            for imported_item in imports:
                # Add imported module/object entity
                if from_module:
                    full_name = f"{from_module}.{imported_item}"
                    module_name = from_module
                else:
                    full_name = imported_item
                    module_name = imported_item
                
                module_entity = {
                    "id": str(uuid.uuid4()),
                    "name": module_name,
                    "type": "module",
                    "properties": {
                        "language": "python",
                        "full_import": full_name,
                        "alias": alias
                    }
                }
                
                result["entities"].append(module_entity)
                
                # Add dependency relationship
                relationship = {
                    "from_id": None,  # Will be resolved later
                    "to_id": None,    # Will be resolved later
                    "from_name": "__current_module__",  # Placeholder
                    "to_name": module_name,
                    "type": "DEPENDS_ON",
                    "properties": {
                        "imported_item": imported_item
                    }
                }
                
                result["relationships"].append(relationship)
        
        # Try to extract using ast for more accurate parsing
        try:
            self._extract_from_python_ast(code, result)
        except SyntaxError:
            # If code has syntax errors, we already extracted what we could with regex
            pass
        except Exception as e:
            logger.warning(f"Error in AST parsing: {e}")
    
    def _extract_from_python_ast(self, code: str, result: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract knowledge elements from Python code using AST.
        
        Args:
            code: Python code content
            result: Dictionary to populate with extracted elements
        """
        tree = ast.parse(code)
        
        # Track current class for method associations
        current_class = None
        
        for node in ast.walk(tree):
            # Extract docstring for decisions/patterns
            docstring = ast.get_docstring(node) if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Module)) else None
            if docstring:
                self._extract_from_docstring(docstring, result)
            
            # Extract class definitions
            if isinstance(node, ast.ClassDef):
                current_class = node.name
                
                # Check for class decorators for potential patterns
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorator_name = decorator.id
                        pattern = {
                            "id": str(uuid.uuid4()),
                            "name": f"{decorator_name} decorator pattern",
                            "description": f"Class {node.name} uses the {decorator_name} decorator pattern",
                            "examples": [f"@{decorator_name}\nclass {node.name}:"]
                        }
                        result["patterns"].append(pattern)
            
            # Extract function definitions and link to classes
            elif isinstance(node, ast.FunctionDef):
                # If we're inside a class, this is a method
                if current_class and node.parent == current_class:
                    # Check for method decorators
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name):
                            decorator_name = decorator.id
                            if decorator_name in ["property", "staticmethod", "classmethod"]:
                                pattern = {
                                    "id": str(uuid.uuid4()),
                                    "name": f"{decorator_name} pattern",
                                    "description": f"Method {node.name} in class {current_class} uses the {decorator_name} pattern",
                                    "examples": [f"@{decorator_name}\ndef {node.name}(self):"]
                                }
                                result["patterns"].append(pattern)
    
    def _extract_from_js_code(self, code: str, result: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract knowledge elements from JavaScript/TypeScript code.
        
        Args:
            code: JavaScript/TypeScript code content
            result: Dictionary to populate with extracted elements
        """
        # Extract classes
        class_pattern = r"class\s+([A-Za-z0-9_]+)(?:\s+extends\s+([A-Za-z0-9_]+))?"
        class_matches = re.finditer(class_pattern, code)
        
        for match in class_matches:
            class_name = match.group(1)
            parent_class = match.group(2) if match.group(2) else ""
            
            # Add class entity
            class_entity = {
                "id": str(uuid.uuid4()),
                "name": class_name,
                "type": "class",
                "properties": {
                    "language": "javascript",
                    "parent_class": parent_class
                }
            }
            
            result["entities"].append(class_entity)
            
            # Add inheritance relationship if parent class exists
            if parent_class:
                relationship = {
                    "from_id": None,  # Will be resolved later
                    "to_id": None,    # Will be resolved later
                    "from_name": class_name,
                    "to_name": parent_class,
                    "type": "INHERITS_FROM",
                    "properties": {}
                }
                
                result["relationships"].append(relationship)
        
        # Extract functions (including arrow functions and methods)
        func_patterns = [
            # Standard functions
            r"function\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)",
            # Arrow functions with explicit name assignment
            r"(?:const|let|var)\s+([A-Za-z0-9_]+)\s*=\s*(?:\(([^)]*)\)|[A-Za-z0-9_]+)\s*=>\s*{",
            # Class methods
            r"(?:async\s+)?([A-Za-z0-9_]+)\s*\(([^)]*)\)\s*{",
            # React functional components (simplified)
            r"(?:const|let|var)\s+([A-Za-z0-9_]+)\s*=\s*\((?:props|{[^}]*})\)\s*=>\s*{?"
        ]
        
        for pattern in func_patterns:
            func_matches = re.finditer(pattern, code)
            
            matches = [m for m in func_matches]
            for match in matches:
                func_name = match.group(1)
                params = match.group(2).strip() if match.group(2) else ""
                
                # Skip if the name is a JS keyword
                if func_name in ["if", "for", "while", "switch", "return", "async"]:
                    continue
                
                # Determine if it's likely a method
                line_start = code[:match.start()].rfind('\n') + 1
                line_before = code[line_start:match.start()].strip()
                
                func_type = "method" if line_before.endswith(':') or pattern == func_patterns[2] else "function"
                
                # Add function entity
                func_entity = {
                    "id": str(uuid.uuid4()),
                    "name": func_name,
                    "type": func_type,
                    "properties": {
                        "language": "javascript",
                        "parameters": params,
                        "is_react_component": "props" in params or pattern == func_patterns[3]
                    }
                }
                
                result["entities"].append(func_entity)
        
        # Extract imports for module dependencies
        import_patterns = [
            # ES6 imports
            r"import\s+(?:{([^}]+)}\s+from\s+)?['\"]([^'\"]+)['\"]",
            # CommonJS require
            r"(?:const|let|var)\s+([A-Za-z0-9_{},:]+)\s*=\s*require\(['\"]([^'\"]+)['\"]"
        ]
        
        for pattern in import_patterns:
            match_groups = [m.groups() for m in re.finditer(pattern, code)]
            for match in match_groups:
                if pattern == import_patterns[0]:  # ES6 import
                    imports = match[0] if match[0] else "*"
                    module_path = match[1]
                else:  # CommonJS require
                    imports = match[0]
                    module_path = match[1]
                
                # Extract module name from path
                module_name = module_path.split('/')[-1]
                
                # Add module entity
                module_entity = {
                    "id": str(uuid.uuid4()),
                    "name": module_name,
                    "type": "module",
                    "properties": {
                        "language": "javascript",
                        "path": module_path,
                        "imported_items": imports
                    }
                }
                
                result["entities"].append(module_entity)
                
                # Add dependency relationship
                relationship = {
                    "from_id": None,  # Will be resolved later
                    "to_id": None,    # Will be resolved later
                    "from_name": "__current_module__",  # Placeholder
                    "to_name": module_name,
                    "type": "DEPENDS_ON",
                    "properties": {}
                }
                
                result["relationships"].append(relationship)
                
        # Look for common JS patterns
        pattern_indicators = {
            "Factory Pattern": r"(?:function|const|var|let)\s+create[A-Z][A-Za-z0-9_]*\s*\(",
            "Singleton Pattern": r"(?:const|var|let)\s+[A-Za-z0-9_]+\s*=\s*\(\s*(?:function)?\s*\(\s*\)\s*{.*?(?:return|this)[^;]*};?\s*\)\(\s*\)",
            "Module Pattern": r"\(\s*function\s*\(\s*\)\s*{.*?return\s*{.*;?\s*\}\)\s*\(\s*\)",
            "Observer Pattern": r"(?:on|addEventListener|subscribe|observe)\s*\(",
            "Prototype Pattern": r"[A-Za-z0-9_]+\.prototype\.",
            "Decorator Pattern": r"function\s+([A-Za-z0-9_]+)\s*\([^)]*\)\s*{\s*(?:var|let|const)?\s*[A-Za-z0-9_]+\s*=\s*[^;]*;\s*[^;]*\.[^;]*\s*=\s*function"
        }
        
        for pattern_name, regex in pattern_indicators.items():
            if re.search(regex, code, re.DOTALL):
                pattern = {
                    "id": str(uuid.uuid4()),
                    "name": pattern_name,
                    "description": f"Code appears to use the {pattern_name}",
                    "examples": []
                }
                result["patterns"].append(pattern)
    
    def _extract_from_generic_code(self, code: str, result: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract knowledge elements from generic code (language-agnostic approach).
        
        Args:
            code: Generic code content
            result: Dictionary to populate with extracted elements
        """
        # Generic function/method pattern
        func_pattern = r"(?:function|def|void|public|private|protected|static|async)?\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)"
        func_matches = re.finditer(func_pattern, code)
        
        for match in func_matches:
            func_name = match.group(1)
            params = match.group(2).strip() if match.group(2) else ""
            
            # Skip if the name is a common keyword
            if func_name in ["if", "for", "while", "switch", "return", "class"]:
                continue
            
            # Add function entity
            func_entity = {
                "id": str(uuid.uuid4()),
                "name": func_name,
                "type": "function",  # Generic type, could be function or method
                "properties": {
                    "parameters": params
                }
            }
            
            result["entities"].append(func_entity)
        
        # Generic class pattern
        class_pattern = r"(?:class|interface|struct|type)\s+([A-Za-z0-9_]+)"
        class_matches = re.finditer(class_pattern, code)
        
        for match in class_matches:
            class_name = match.group(1)
            
            # Add class entity
            class_entity = {
                "id": str(uuid.uuid4()),
                "name": class_name,
                "type": "class",  # Generic type for any class-like structure
                "properties": {}
            }
            
            result["entities"].append(class_entity)
        
        # Generic variable declaration pattern
        var_pattern = r"(?:var|let|const|int|float|double|string|boolean|char|[A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*(?:=|:)"
        var_matches = re.finditer(var_pattern, code)
        
        matches = [m for m in var_matches]
        for match in matches:
            var_name = match.group(1)
            
            # Skip if the name is a common keyword
            if var_name in ["if", "for", "while", "switch", "return", "function", "class"]:
                continue
            
            # Add variable entity
            var_entity = {
                "id": str(uuid.uuid4()),
                "name": var_name,
                "type": "variable",
                "properties": {}
            }
            
            result["entities"].append(var_entity)
            
        # Look for common design patterns
        pattern_indicators = {
            "Factory Pattern": r"(?:create|factory|make)[A-Z][A-Za-z0-9_]*\s*\(",
            "Singleton Pattern": r"(?:getInstance|shared|default)\s*\(\s*\)",
            "Observer Pattern": r"(?:notify|update|subscribe|observer)",
            "Strategy Pattern": r"(?:strategy|algorithm|behavior)",
            "Adapter Pattern": r"(?:adapter|adapt|wrapper)",
            "Decorator Pattern": r"(?:decorator|decorate|wrap)",
            "Repository Pattern": r"(?:repository|repo|dao|store)\.",
            "MVC Pattern": r"(?:model|view|controller)"
        }
        
        for pattern_name, regex in pattern_indicators.items():
            if re.search(regex, code, re.IGNORECASE):
                pattern = {
                    "id": str(uuid.uuid4()),
                    "name": pattern_name,
                    "description": f"Code appears to use the {pattern_name}",
                    "examples": []
                }
                result["patterns"].append(pattern)
    
    def _extract_from_markdown(self, markdown: str, result: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract knowledge elements from markdown content.
        
        This function specifically looks for:
        1. Decision documentation in markdown headings
        2. Code blocks that might contain entities
        3. Explicit pattern documentation
        
        Args:
            markdown: Markdown content
            result: Dictionary to populate with extracted elements
        """
        # Extract decision headings and their content
        decision_headers = [
            "## Decision", 
            "### Decision",
            "## Architectural Decision",
            "### Architectural Decision",
            "## Design Decision",
            "### Design Decision"
        ]
        
        for header in decision_headers:
            if header in markdown:
                # Find the section content (until the next heading or end of text)
                start_idx = markdown.find(header) + len(header)
                next_heading_idx = markdown.find("#", start_idx)
                end_idx = next_heading_idx if next_heading_idx != -1 else len(markdown)
                
                decision_content = markdown[start_idx:end_idx].strip()
                
                # Extract title, reasoning, alternatives
                title_match = re.search(r"(?:Title|Name):\s*([^\n]+)", decision_content)
                title = title_match.group(1) if title_match else "Unnamed Decision"
                
                reasoning_match = re.search(r"(?:Reasoning|Rationale|Justification):\s*([^\n]+(?:\n(?!\#\#)[^\n]+)*)", decision_content)
                reasoning = reasoning_match.group(1) if reasoning_match else ""
                
                alternatives_match = re.search(r"(?:Alternatives|Other Options):\s*([^\n]+(?:\n(?!\#\#)[^\n]+)*)", decision_content)
                alternatives_text = alternatives_match.group(1) if alternatives_match else ""
                alternatives = [alt.strip() for alt in alternatives_text.split(",")]
                
                decision = {
                    "id": str(uuid.uuid4()),
                    "title": title,
                    "description": decision_content[:100] + "..." if len(decision_content) > 100 else decision_content,
                    "reasoning": reasoning,
                    "alternatives": alternatives,
                    "affects": []
                }
                
                result["decisions"].append(decision)
        
        # Extract pattern documentation
        pattern_headers = [
            "## Pattern",
            "### Pattern",
            "## Design Pattern",
            "### Design Pattern"
        ]
        
        for header in pattern_headers:
            if header in markdown:
                # Find the section content (until the next heading or end of text)
                start_idx = markdown.find(header) + len(header)
                next_heading_idx = markdown.find("#", start_idx)
                end_idx = next_heading_idx if next_heading_idx != -1 else len(markdown)
                
                pattern_content = markdown[start_idx:end_idx].strip()
                
                # Extract name and description
                name_match = re.search(r"(?:Name|Title):\s*([^\n]+)", pattern_content)
                name = name_match.group(1) if name_match else "Unnamed Pattern"
                
                desc_match = re.search(r"(?:Description|Summary):\s*([^\n]+(?:\n(?!\#\#)[^\n]+)*)", pattern_content)
                description = desc_match.group(1) if desc_match else pattern_content
                
                pattern = {
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "description": description[:200] + "..." if len(description) > 200 else description,
                    "examples": []
                }
                
                result["patterns"].append(pattern)
        
        # Extract code blocks that might contain entities
        code_blocks = re.finditer(r"```(?:python|javascript|js|typescript|ts|java|csharp|cs|c\+\+|cpp)?\n(.*?)```", markdown, re.DOTALL)
        
        for block in code_blocks:
            code_content = block.group(1)
            if "```python" in block.group(0):
                self._extract_from_python_code(code_content, result)
            elif any(lang in block.group(0) for lang in ["```javascript", "```js", "```typescript", "```ts"]):
                self._extract_from_js_code(code_content, result)
            else:
                self._extract_from_generic_code(code_content, result)
    
    def _extract_from_docstring(self, docstring: str, result: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract knowledge from Python docstrings.
        
        Args:
            docstring: Python docstring content
            result: Dictionary to populate with extracted elements
        """
        # Look for design decisions in docstrings
        decision_patterns = [
            r"Design Decision:?\s*([^\n]+)",
            r"Decision:?\s*([^\n]+)",
            r"Architectural Decision:?\s*([^\n]+)"
        ]
        
        for pattern in decision_patterns:
            decision_match = re.search(pattern, docstring, re.IGNORECASE)
            if decision_match:
                title = decision_match.group(1).strip()
                
                # Try to find reasoning
                reasoning_match = re.search(r"Reasoning:?\s*([^\n]+)", docstring, re.IGNORECASE)
                reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
                
                decision = {
                    "id": str(uuid.uuid4()),
                    "title": title,
                    "description": docstring[:100] + "..." if len(docstring) > 100 else docstring,
                    "reasoning": reasoning,
                    "alternatives": []
                }
                
                # Look for alternatives
                alternatives_match = re.search(r"Alternatives:?\s*([^\n]+)", docstring, re.IGNORECASE)
                if alternatives_match:
                    alternatives = alternatives_match.group(1).split(",")
                    decision["alternatives"] = [alt.strip() for alt in alternatives]
                
                result["decisions"].append(decision)
                break  # Only extract one decision per docstring
        
        # Look for design patterns in docstrings
        pattern_patterns = [
            r"(?:Design )?Pattern:?\s*([^\n]+)",
            r"Uses Pattern:?\s*([^\n]+)",
            r"Implements Pattern:?\s*([^\n]+)"
        ]
        
        for pattern in pattern_patterns:
            pattern_match = re.search(pattern, docstring, re.IGNORECASE)
            if pattern_match:
                name = pattern_match.group(1).strip()
                
                # Try to find description
                description_match = re.search(r"Description:?\s*([^\n]+)", docstring, re.IGNORECASE)
                description = description_match.group(1).strip() if description_match else ""
                
                pattern_dict = {
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "description": description or f"Implements the {name} pattern",
                    "examples": []
                }
                
                result["patterns"].append(pattern_dict)
                break  # Only extract one pattern per docstring
                
    def _find_entity_by_name(self, entities: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
        """
        Find an entity by name in a list of entities.
        
        Args:
            entities: List of entity dictionaries
            name: Entity name to find
            
        Returns:
            Entity dictionary or None if not found
        """
        for entity in entities:
            if entity["name"].lower() == name.lower():
                return entity
        return None
    
    def resolve_relationships(self, result: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Resolve entity IDs in relationships based on entity names.
        
        Args:
            result: Dictionary with extracted knowledge elements
        """
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])
        
        for relationship in relationships:
            from_name = relationship.get("from_name")
            to_name = relationship.get("to_name")
            
            if from_name and from_name != "__current_module__":
                from_entity = self._find_entity_by_name(entities, from_name)
                if from_entity:
                    relationship["from_id"] = from_entity["id"]
            
            if to_name:
                to_entity = self._find_entity_by_name(entities, to_name)
                if to_entity:
                    relationship["to_id"] = to_entity["id"]
        
        # Remove relationships with unresolved entities
        result["relationships"] = [
            rel for rel in relationships 
            if (rel.get("from_id") is not None or rel["from_name"] == "__current_module__") 
               and rel.get("to_id") is not None
        ]
    
    def deduplicate_knowledge(self, result: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Remove duplicate knowledge elements.
        
        Args:
            result: Dictionary with extracted knowledge elements
        """
        # Deduplicate entities
        unique_entities = {}
        for entity in result.get("entities", []):
            key = f"{entity['type']}:{entity['name']}"
            if key not in unique_entities:
                unique_entities[key] = entity
        result["entities"] = list(unique_entities.values())
        
        # Deduplicate decisions
        unique_decisions = {}
        for decision in result.get("decisions", []):
            if decision["title"] not in unique_decisions:
                unique_decisions[decision["title"]] = decision
        result["decisions"] = list(unique_decisions.values())
        
        # Deduplicate patterns
        unique_patterns = {}
        for pattern in result.get("patterns", []):
            if pattern["name"] not in unique_patterns:
                unique_patterns[pattern["name"]] = pattern
        result["patterns"] = list(unique_patterns.values())
        
        # Deduplicate relationships
        unique_relationships = {}
        for rel in result.get("relationships", []):
            key = f"{rel['from_name']}:{rel['type']}:{rel['to_name']}"
            if key not in unique_relationships:
                unique_relationships[key] = rel
        result["relationships"] = list(unique_relationships.values())
        