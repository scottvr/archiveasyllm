"""
Codebase analyzer for initializing the knowledge graph from existing code.
"""
# crumb: analyzer\codebase.py
import os
import re
from typing import Dict, Any, List, Optional, Set, Tuple
import logging
from pathlib import Path
import ast
import json

from archiveasy.memory.graph import KnowledgeGraph
from archiveasy.memory.vector import VectorStore
from archiveasy.memory.extractor import KnowledgeExtractor

logger = logging.getLogger(__name__)

class CodebaseAnalyzer:
    """
    Analyzes an existing codebase to initialize the knowledge graph.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, vector_store: VectorStore):
        """
        Initialize the codebase analyzer.
        
        Args:
            knowledge_graph: Knowledge graph for storing extracted information
            vector_store: Vector store for semantic search
        """
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
        self.extractor = KnowledgeExtractor()
        
        # Track processed files to avoid duplicates
        self.processed_files = set()
        
        # Map file extensions to languages
        self.language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'jsx',
            '.tsx': 'tsx',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.rs': 'rust',
            '.md': 'markdown',
        }
        
    def analyze_codebase(self, codebase_path: str, project_id: str, excluded_dirs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze a codebase and populate the knowledge graph.
        
        Args:
            codebase_path: Path to the codebase root directory
            project_id: Project identifier
            excluded_dirs: List of directories to exclude (e.g., node_modules, venv)
            
        Returns:
            Summary of analysis results
        """
        if excluded_dirs is None:
            excluded_dirs = ['node_modules', 'venv', '.venv', 'env', '__pycache__', '.git', 'dist', 'build']
        
        # Reset processed files
        self.processed_files = set()
        
        # Initialize the project in the knowledge graph
        self.knowledge_graph.initialize_project(project_id)
        
        # Collect statistics
        stats = {
            "files_processed": 0,
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "decisions_extracted": 0,
            "patterns_extracted": 0,
            "languages": {}
        }
        
        print(f"Analyzing codebase at: {codebase_path}")

        # Walk the directory structure
        codebase_path = os.path.abspath(codebase_path)
        for root, dirs, files in os.walk(codebase_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            
            # Process each file
            for file in files:
                file_path = os.path.join(root, file)
                print(f"found file: {file_path}") 
                # Skip if already processed
                if file_path in self.processed_files:
                    continue
                
                # Get file extension
                _, ext = os.path.splitext(file)
                
                # Skip if unsupported file type
                if ext not in self.language_map:
                    continue
                
                # Mark as processed
                self.processed_files.add(file_path)
                
                # Process file based on type
                try:
                    language = self.language_map[ext]
                    
                    # Update language stats
                    if language not in stats["languages"]:
                        stats["languages"][language] = 0
                    stats["languages"][language] += 1
                    
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Extract knowledge based on file type
                    file_stats = self._process_file(file_path, content, language, project_id)
                    
                    # Update statistics
                    stats["files_processed"] += 1
                    stats["entities_extracted"] += file_stats["entities"]
                    stats["relationships_extracted"] += file_stats["relationships"]
                    stats["decisions_extracted"] += file_stats["decisions"]
                    stats["patterns_extracted"] += file_stats["patterns"]
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
        
        # Extract module-level relationships
        self._extract_module_relationships(project_id)
        
        return stats
    
    def _process_file(self, file_path: str, content: str, language: str, project_id: str) -> Dict[str, int]:
        """
        Process a single file and extract knowledge.
        
        Args:
            file_path: Path to the file
            content: File content
            language: Programming language
            project_id: Project identifier
            
        Returns:
            Statistics for this file
        """
        file_stats = {
            "entities": 0,
            "relationships": 0,
            "decisions": 0,
            "patterns": 0
        }
        
        # Create a file entity
        file_entity = {
            "id": self._hash_path(file_path),
            "name": os.path.basename(file_path),
            "type": "file",
            "properties": {
                "path": file_path,
                "language": language
            }
        }
        
        with self.knowledge_graph.driver.session() as session:
            # Check if file entity already exists
            result = session.run(
                """
                MATCH (f:Entity {id: $id, project_id: $project_id})
                RETURN count(f) as count
                """,
                id=file_entity["id"],
                project_id=project_id
            )
    
            if result.single()["count"] == 0:
                # Entity doesn't exist, so create it
                print(f"Creating entity {file_path}")
                session.run(
                    """
                    MATCH (p:Project {id: $project_id})
                    MERGE (f:Entity {id: $id, project_id: $project_id})
                    SET f.type = $type, 
                        f.name = $name, 
                        f.updated_at = datetime(),
                        f += $properties
                    MERGE (f)-[:BELONGS_TO]->(p)
                    """,
                    id=file_entity["id"],
                    project_id=project_id,
                    type=file_entity["type"],
                    name=file_entity["name"],
                    properties=file_entity["properties"]
                )
            else:
                print(f"Skipping entity ${file_path}")

        file_stats["entities"] += 1
        
        # Extract knowledge based on language
        knowledge = {}
        
        if language == "python":
            knowledge = self._extract_from_python(content, file_path)
        elif language in ["javascript", "typescript", "jsx", "tsx"]:
            knowledge = self._extract_from_js(content, file_path)
        elif language == "markdown":
            knowledge = self._extract_from_markdown(content, file_path)
        else:
            # Use generic extraction for other languages
            knowledge = self._extract_generic(content, file_path)
        
        # Link entities to this file
        for entity in knowledge.get("entities", []):
            entity_id = entity.get("id")
            
            if entity_id:
                with self.knowledge_graph.driver.session() as session:
                    session.run(
                        """
                        MATCH (f:Entity {id: $file_id, project_id: $project_id})
                        MATCH (e:Entity {id: $entity_id, project_id: $project_id})
                        MERGE (e)-[:DEFINED_IN]->(f)
                        """,
                        file_id=file_entity["id"],
                        entity_id=entity_id,
                        project_id=project_id
                    )
        
        # Store in knowledge graph
        self.knowledge_graph.store(knowledge, project_id)
        
        # Store in vector store for semantic search
        self.vector_store.store(content, [], project_id)
        
        # Update statistics
        file_stats["entities"] += len(knowledge.get("entities", []))
        file_stats["relationships"] += len(knowledge.get("relationships", []))
        file_stats["decisions"] += len(knowledge.get("decisions", []))
        file_stats["patterns"] += len(knowledge.get("patterns", []))
        
        return file_stats
    
    def _extract_from_python(self, content: str, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract knowledge from Python code using AST.
        
        Args:
            content: Python code content
            file_path: Path to the file
            
        Returns:
            Extracted knowledge
        """
        result = {
            "entities": [],
            "relationships": [],
            "decisions": [],
            "patterns": []
        }
        
        # Use the KnowledgeExtractor for standard pattern extraction
        standard_knowledge = self.extractor._extract_from_python_code(content, {})
        result.update(standard_knowledge)
        
        try:
            # Parse the AST for more detailed analysis
            tree = ast.parse(content)
            
            # Extract module-level imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        module_entity = {
                            "id": self._hash_name(name.name),
                            "name": name.name,
                            "type": "module",
                            "properties": {
                                "language": "python",
                                "asname": name.asname
                            }
                        }
                        result["entities"].append(module_entity)
                        
                        # Add dependency relationship
                        relationship = {
                            "from_id": self._hash_path(file_path),
                            "to_id": self._hash_name(name.name),
                            "type": "IMPORTS",
                            "properties": {}
                        }
                        result["relationships"].append(relationship)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_entity = {
                            "id": self._hash_name(node.module),
                            "name": node.module,
                            "type": "module",
                            "properties": {
                                "language": "python"
                            }
                        }
                        result["entities"].append(module_entity)
                        
                        # Add dependency relationship
                        relationship = {
                            "from_id": self._hash_path(file_path),
                            "to_id": self._hash_name(node.module),
                            "type": "IMPORTS_FROM",
                            "properties": {}
                        }
                        result["relationships"].append(relationship)
                        
                        # Add imported names
                        for name in node.names:
                            imported_entity = {
                                "id": self._hash_name(f"{node.module}.{name.name}"),
                                "name": name.name,
                                "type": "imported_entity",
                                "properties": {
                                    "language": "python",
                                    "module": node.module,
                                    "asname": name.asname
                                }
                            }
                            result["entities"].append(imported_entity)
                            
                            # Relationship between imported entity and module
                            relationship = {
                                "from_id": self._hash_name(f"{node.module}.{name.name}"),
                                "to_id": self._hash_name(node.module),
                                "type": "DEFINED_IN",
                                "properties": {}
                            }
                            result["relationships"].append(relationship)
                
                # Extract class definitions with inheritance
                elif isinstance(node, ast.ClassDef):
                    class_entity = {
                        "id": self._hash_name(node.name),
                        "name": node.name,
                        "type": "class",
                        "properties": {
                            "language": "python",
                            "docstring": ast.get_docstring(node) or ""
                        }
                    }
                    result["entities"].append(class_entity)
                    
                    # Extract inheritance relationships
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            relationship = {
                                "from_id": self._hash_name(node.name),
                                "to_id": self._hash_name(base.id),
                                "type": "INHERITS_FROM",
                                "properties": {}
                            }
                            result["relationships"].append(relationship)
                    
                    # Extract methods
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            method_entity = {
                                "id": self._hash_name(f"{node.name}.{child.name}"),
                                "name": child.name,
                                "type": "method",
                                "properties": {
                                    "language": "python",
                                    "docstring": ast.get_docstring(child) or "",
                                    "is_special": child.name.startswith("__") and child.name.endswith("__")
                                }
                            }
                            result["entities"].append(method_entity)
                            
                            # Relationship between method and class
                            relationship = {
                                "from_id": self._hash_name(f"{node.name}.{child.name}"),
                                "to_id": self._hash_name(node.name),
                                "type": "DEFINED_IN",
                                "properties": {}
                            }
                            result["relationships"].append(relationship)
                
                # Extract function definitions
                elif isinstance(node, ast.FunctionDef) and not isinstance(node.parent, ast.ClassDef):
                    func_entity = {
                        "id": self._hash_name(node.name),
                        "name": node.name,
                        "type": "function",
                        "properties": {
                            "language": "python",
                            "docstring": ast.get_docstring(node) or ""
                        }
                    }
                    result["entities"].append(func_entity)
        
        except SyntaxError as e:
            logger.warning(f"Syntax error in Python file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error analyzing Python file {file_path}: {e}")
        
        # Extract docstring comments for design decisions and patterns
        self._extract_from_docstrings(content, result)
        
        return result
    
    def _extract_from_js(self, content: str, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract knowledge from JavaScript/TypeScript code.
        
        Args:
            content: JavaScript code content
            file_path: Path to the file
            
        Returns:
            Extracted knowledge
        """
        # For simplicity, use the existing extractor
        result = {}
        result = self.extractor._extract_from_js_code(content, result)
        
        # Additionally, look for JSDoc comments for design decisions
        jsflow_pattern = r"/\*\*\s*\n(?:\s*\*\s*(?:@[a-zA-Z]+\s+[^\n]*)?\n)+\s*\*/"
        jsdoc_comments = re.finditer(jsflow_pattern, content)
        
        for comment in jsdoc_comments:
            comment_text = comment.group(0)
            
            # Extract design decisions
            if "@decision" in comment_text:
                decision_match = re.search(r"@decision\s+([^\n]+)", comment_text)
                reasoning_match = re.search(r"@reasoning\s+([^\n]+)", comment_text)
                
                if decision_match:
                    decision = {
                        "id": self._hash_text(comment_text),
                        "title": decision_match.group(1),
                        "description": comment_text,
                        "reasoning": reasoning_match.group(1) if reasoning_match else "",
                        "alternatives": []
                    }
                    
                    # Look for alternatives
                    alternatives_match = re.search(r"@alternatives\s+([^\n]+)", comment_text)
                    if alternatives_match:
                        alternatives = alternatives_match.group(1).split(",")
                        decision["alternatives"] = [alt.strip() for alt in alternatives]
                    
                    result.setdefault("decisions", []).append(decision)
            
            # Extract design patterns
            if "@pattern" in comment_text:
                pattern_match = re.search(r"@pattern\s+([^\n]+)", comment_text)
                description_match = re.search(r"@description\s+([^\n]+)", comment_text)
                
                if pattern_match:
                    pattern = {
                        "id": self._hash_text(comment_text),
                        "name": pattern_match.group(1),
                        "description": description_match.group(1) if description_match else comment_text,
                        "examples": []
                    }
                    
                    result.setdefault("patterns", []).append(pattern)
        
        return result
    
    def _extract_from_markdown(self, content: str, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract knowledge from Markdown documents.
        
        Args:
            content: Markdown content
            file_path: Path to the file
            
        Returns:
            Extracted knowledge
        """
        # Use the existing extractor for markdown
        result = {}
        self.extractor._extract_from_markdown(content, result)
        
        # Look for architecture decision records (ADRs)
        if "decision" in file_path.lower() or "adr" in file_path.lower():
            # Extract title from first heading
            title_match = re.search(r"#\s+(.+)$", content, re.MULTILINE)
            title = title_match.group(1) if title_match else os.path.basename(file_path)
            
            # Extract status if available
            status_match = re.search(r"status:?\s*(.+)$", content, re.MULTILINE | re.IGNORECASE)
            status = status_match.group(1) if status_match else "Unknown"
            
            # Extract context, decision, and consequences sections
            context_match = re.search(r"#+\s+Context.*?\n(.*?)(?=#+\s+|$)", content, re.DOTALL)
            decision_match = re.search(r"#+\s+Decision.*?\n(.*?)(?=#+\s+|$)", content, re.DOTALL)
            consequences_match = re.search(r"#+\s+Consequences.*?\n(.*?)(?=#+\s+|$)", content, re.DOTALL)
            
            context = context_match.group(1).strip() if context_match else ""
            decision_text = decision_match.group(1).strip() if decision_match else ""
            consequences = consequences_match.group(1).strip() if consequences_match else ""
            
            # Create a comprehensive decision record
            decision = {
                "id": self._hash_path(file_path),
                "title": title,
                "description": f"Status: {status}\n\nContext: {context}",
                "reasoning": decision_text,
                "alternatives": [],
                "consequences": consequences
            }
            
            # Look for alternatives section
            alternatives_match = re.search(r"#+\s+Alternatives.*?\n(.*?)(?=#+\s+|$)", content, re.DOTALL)
            if alternatives_match:
                # Extract bullet points as alternatives
                alt_text = alternatives_match.group(1)
                alt_items = re.findall(r"[*-]\s+(.+)$", alt_text, re.MULTILINE)
                decision["alternatives"] = alt_items
            
            result.setdefault("decisions", []).append(decision)
        
        return result
    
    def _extract_generic(self, content: str, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract knowledge from generic code files.
        
        Args:
            content: Code content
            file_path: Path to the file
            
        Returns:
            Extracted knowledge
        """
        # Use the existing extractor for generic code
        result = {}
        self.extractor._extract_from_generic_code(content, result)
        
        # Look for comments that might indicate design decisions or patterns
        # Multi-line comment pattern (works for many languages)
        multiline_comment_pattern = r"/\*\*?(.*?)\*/"
        multiline_comments = re.finditer(multiline_comment_pattern, content, re.DOTALL)
        
        for comment in multiline_comments:
            comment_text = comment.group(1)
            
            # Check for decision markers
            decision_markers = ["DECISION:", "Design Decision:", "Architectural Decision:"]
            for marker in decision_markers:
                if marker in comment_text:
                    parts = comment_text.split(marker, 1)
                    if len(parts) > 1:
                        decision_text = parts[1].strip()
                        
                        # Try to split into title and reasoning
                        title_end = decision_text.find("\n")
                        if title_end > 0:
                            title = decision_text[:title_end].strip()
                            reasoning = decision_text[title_end:].strip()
                        else:
                            title = decision_text
                            reasoning = ""
                        
                        decision = {
                            "id": self._hash_text(comment_text),
                            "title": title,
                            "description": comment_text,
                            "reasoning": reasoning,
                            "alternatives": []
                        }
                        
                        result.setdefault("decisions", []).append(decision)
            
            # Check for pattern markers
            pattern_markers = ["PATTERN:", "Design Pattern:", "Pattern:"]
            for marker in pattern_markers:
                if marker in comment_text:
                    parts = comment_text.split(marker, 1)
                    if len(parts) > 1:
                        pattern_text = parts[1].strip()
                        
                        # Try to split into name and description
                        name_end = pattern_text.find("\n")
                        if name_end > 0:
                            name = pattern_text[:name_end].strip()
                            description = pattern_text[name_end:].strip()
                        else:
                            name = pattern_text
                            description = ""
                        
                        pattern = {
                            "id": self._hash_text(comment_text),
                            "name": name,
                            "description": description,
                            "examples": []
                        }
                        
                        result.setdefault("patterns", []).append(pattern)
        
        return result
    
    def _extract_from_docstrings(self, content: str, result: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract knowledge from Python docstrings.
        
        Args:
            content: Python code content
            result: Result dictionary to update
        """
        # Look for design decisions in docstrings
        decision_pattern = r'""".*?Design Decision:?\s*([^\n]+).*?"""'
        decision_matches = re.finditer(decision_pattern, content, re.DOTALL)
        
        for match in decision_matches:
            full_text = match.group(0)
            title = match.group(1).strip()
            
            # Try to find reasoning
            reasoning_match = re.search(r"Reasoning:?\s*([^\n]+)", full_text)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            
            decision = {
                "id": self._hash_text(full_text),
                "title": title,
                "description": full_text,
                "reasoning": reasoning,
                "alternatives": []
            }
            
            # Look for alternatives
            alternatives_match = re.search(r"Alternatives:?\s*([^\n]+)", full_text)
            if alternatives_match:
                alternatives = alternatives_match.group(1).split(",")
                decision["alternatives"] = [alt.strip() for alt in alternatives]
            
            result.setdefault("decisions", []).append(decision)
        
        # Look for design patterns in docstrings
        pattern_pattern = r'""".*?(?:Design )?Pattern:?\s*([^\n]+).*?"""'
        pattern_matches = re.finditer(pattern_pattern, content, re.DOTALL)
        
        for match in pattern_matches:
            full_text = match.group(0)
            name = match.group(1).strip()
            
            # Try to find description
            description_match = re.search(r"Description:?\s*([^\n]+)", full_text)
            description = description_match.group(1).strip() if description_match else ""
            
            pattern = {
                "id": self._hash_text(full_text),
                "name": name,
                "description": description or full_text,
                "examples": []
            }
            
            result.setdefault("patterns", []).append(pattern)
    
    def _extract_module_relationships(self, project_id: str) -> None:
        """
        Extract relationships between modules based on imports.
        
        Args:
            project_id: Project identifier
        """
        with self.knowledge_graph.driver.session() as session:
            # Find all import relationships
            session.run(
                """
                MATCH (importer:Entity {project_id: $project_id})-[:IMPORTS]->(imported:Entity {project_id: $project_id})
                MERGE (importer)-[:DEPENDS_ON]->(imported)
                """,
                project_id=project_id
            )
            
            # Find all import-from relationships
            session.run(
                """
                MATCH (importer:Entity {project_id: $project_id})-[:IMPORTS_FROM]->(module:Entity {project_id: $project_id})
                MATCH (entity:Entity {project_id: $project_id})-[:DEFINED_IN]->(module)
                MERGE (importer)-[:DEPENDS_ON]->(entity)
                """,
                project_id=project_id
            )
    
    def _hash_path(self, path: str) -> str:
        """
        Create a consistent hash for a file path.
        
        Args:
            path: File path
            
        Returns:
            Hashed identifier
        """
        import hashlib
        return f"file-{hashlib.md5(path.encode()).hexdigest()}"
    
    def _hash_name(self, name: str) -> str:
        """
        Create a consistent hash for an entity name.
        
        Args:
            name: Entity name
            
        Returns:
            Hashed identifier
        """
        import hashlib
        return f"entity-{hashlib.md5(name.encode()).hexdigest()}"
    
    def _hash_text(self, text: str) -> str:
        """
        Create a consistent hash for text content.
        
        Args:
            text: Text content
            
        Returns:
            Hashed identifier
        """
        import hashlib
        return f"content-{hashlib.md5(text.encode()).hexdigest()}"
