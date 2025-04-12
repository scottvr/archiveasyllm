"""
Knowledge extractor module for identifying architectural patterns and decisions in LLM responses.
"""
from typing import Dict, Any, List, Optional
import re
import uuid
import json

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
        - "The architecture will use