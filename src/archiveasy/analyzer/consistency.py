"""
Consistency checker for identifying deviations from established patterns.
"""
# crumb: analyzer\consistency.py
from typing import Dict, Any, List, Optional
import re
from archiveasy.memory.graph import KnowledgeGraph
from archiveasy.memory.vector import VectorStore
from archiveasy.memory.extractor import KnowledgeExtractor

class ConsistencyChecker:
    """
    Analyzes new LLM responses against established knowledge to identify
    inconsistencies or deviations from patterns.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, vector_store: VectorStore):
        """
        Initialize the consistency checker.
        
        Args:
            knowledge_graph: Knowledge graph for architectural knowledge
            vector_store: Vector store for semantic search
        """
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
        self.extractor = KnowledgeExtractor()
    
    def check(self, response: str, project_id: str, artifacts: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Check response for consistency with existing knowledge.
        
        Args:
            response: Text response from LLM
            project_id: Project identifier
            artifacts: Optional list of artifacts (code, diagrams, etc.)
            
        Returns:
            List of identified inconsistencies
        """
        artifacts = artifacts or []
        inconsistencies = []
        
        # Extract knowledge from the new response
        new_knowledge = self.extractor.extract(response, artifacts)
        
        # Check for naming inconsistencies
        naming_issues = self._check_naming_conventions(new_knowledge, project_id)
        inconsistencies.extend(naming_issues)
        
        # Check for architectural pattern inconsistencies
        pattern_issues = self._check_pattern_consistency(new_knowledge, project_id)
        inconsistencies.extend(pattern_issues)
        
        # Check for design decision contradictions
        decision_issues = self._check_decision_consistency(new_knowledge, project_id)
        inconsistencies.extend(decision_issues)
        
        # Check for structural inconsistencies
        structure_issues = self._check_structural_consistency(new_knowledge, project_id)
        inconsistencies.extend(structure_issues)
        
        return inconsistencies
    
    def _check_naming_conventions(self, new_knowledge: Dict[str, Any], project_id: str) -> List[Dict[str, Any]]:
        """
        Check for inconsistencies in naming conventions.
        
        Args:
            new_knowledge: Extracted knowledge from new response
            project_id: Project identifier
            
        Returns:
            List of naming inconsistencies
        """
        issues = []
        
        # Get established naming patterns from knowledge graph
        naming_patterns = self._get_naming_patterns(project_id)
        
        if not naming_patterns:
            return []  # No established patterns yet
        
        # Check each entity against established patterns
        for entity in new_knowledge.get("entities", []):
            entity_type = entity.get("type", "")
            entity_name = entity.get("name", "")
            
            # Skip if no name or no type
            if not entity_name or not entity_type:
                continue
            
            # Check if this entity type has established patterns
            if entity_type in naming_patterns:
                pattern = naming_patterns[entity_type]
                
                # Check if name matches pattern
                if not re.match(pattern["regex"], entity_name):
                    issues.append({
                        "type": "naming_convention",
                        "severity": "warning",
                        "entity_type": entity_type,
                        "entity_name": entity_name,
                        "expected_pattern": pattern["description"],
                        "message": f"The {entity_type} name '{entity_name}' doesn't follow the established naming convention: {pattern['description']}"
                    })
        
        return issues
    
    def _get_naming_patterns(self, project_id: str) -> Dict[str, Dict[str, str]]:
        """
        Get established naming patterns from the knowledge graph.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Dictionary mapping entity types to naming pattern info
        """
        patterns = {}
        
        with self.knowledge_graph.driver.session() as session:
            # Query for naming patterns
            result = session.run(
                """
                MATCH (p:Pattern {project_id: $project_id})
                WHERE p.name CONTAINS 'naming' OR p.name CONTAINS 'convention'
                RETURN p.name as name, p.description as description, p.properties as properties
                """,
                project_id=project_id
            )
            
            for record in result:
                pattern_name = record["name"]
                description = record["description"]
                properties = record.get("properties", {})
                
                # Extract entity type from pattern name
                entity_type_match = re.search(r"([a-zA-Z]+)(?:\s+naming|\s+convention)", pattern_name.lower())
                if entity_type_match:
                    entity_type = entity_type_match.group(1)
                    
                    # Extract regex pattern if available, or generate one from description
                    regex = properties.get("regex", "")
                    if not regex:
                        if "camelCase" in description:
                            regex = r"^[a-z][a-zA-Z0-9]*$"
                        elif "PascalCase" in description:
                            regex = r"^[A-Z][a-zA-Z0-9]*$"
                        elif "snake_case" in description:
                            regex = r"^[a-z][a-z0-9_]*$"
                        elif "SCREAMING_SNAKE_CASE" in description:
                            regex = r"^[A-Z][A-Z0-9_]*$"
                        elif "kebab-case" in description:
                            regex = r"^[a-z][a-z0-9\-]*$"
                    
                    if regex:
                        patterns[entity_type] = {
                            "description": description,
                            "regex": regex
                        }
        
        return patterns
    
    def _check_pattern_consistency(self, new_knowledge: Dict[str, Any], project_id: str) -> List[Dict[str, Any]]:
        """
        Check for inconsistencies with established design patterns.
        
        Args:
            new_knowledge: Extracted knowledge from new response
            project_id: Project identifier
            
        Returns:
            List of pattern inconsistencies
        """
        issues = []
        
        # Get established patterns from knowledge graph
        established_patterns = self._get_established_patterns(project_id)
        
        # Check for new patterns that conflict with established ones
        for pattern in new_knowledge.get("patterns", []):
            pattern_name = pattern.get("name", "").lower()
            
            # Skip unnamed patterns
            if not pattern_name:
                continue
            
            # Look for conflicting established patterns
            for est_pattern in established_patterns:
                est_name = est_pattern.get("name", "").lower()
                
                # If names are similar but descriptions differ significantly
                if self._similar_names(pattern_name, est_name) and not self._similar_descriptions(
                    pattern.get("description", ""),
                    est_pattern.get("description", "")
                ):
                    issues.append({
                        "type": "pattern_conflict",
                        "severity": "error",
                        "pattern_name": pattern_name,
                        "established_pattern": est_name,
                        "message": f"The pattern '{pattern_name}' conflicts with established pattern '{est_name}'"
                    })
        
        return issues
    
    def _get_established_patterns(self, project_id: str) -> List[Dict[str, Any]]:
        """
        Get established patterns from the knowledge graph.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of established pattern dictionaries
        """
        patterns = []
        
        with self.knowledge_graph.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Pattern {project_id: $project_id})
                RETURN p.id as id, p.name as name, p.description as description
                """,
                project_id=project_id
            )
            
            for record in result:
                patterns.append({
                    "id": record["id"],
                    "name": record["name"],
                    "description": record["description"]
                })
        
        return patterns
    
    def _similar_names(self, name1: str, name2: str) -> bool:
        """
        Check if two pattern names are similar.
        
        Args:
            name1: First pattern name
            name2: Second pattern name
            
        Returns:
            True if names are similar
        """
        # Simple similarity check - could be more sophisticated
        name1 = name1.lower().replace(' ', '').replace('-', '').replace('_', '')
        name2 = name2.lower().replace(' ', '').replace('-', '').replace('_', '')
        
        return name1 == name2 or name1 in name2 or name2 in name1
    
    def _similar_descriptions(self, desc1: str, desc2: str) -> bool:
        """
        Check if two pattern descriptions are similar.
        
        Args:
            desc1: First pattern description
            desc2: Second pattern description
            
        Returns:
            True if descriptions are similar
        """
        # Simple similarity check - in a real system, use vector similarity
        common_words = set(desc1.lower().split()) & set(desc2.lower().split())
        total_words = set(desc1.lower().split()) | set(desc2.lower().split())
        
        if not total_words:
            return False
            
        # Jaccard similarity
        similarity = len(common_words) / len(total_words)
        
        return similarity > 0.3  # Arbitrary threshold
    
    def _check_decision_consistency(self, new_knowledge: Dict[str, Any], project_id: str) -> List[Dict[str, Any]]:
        """
        Check for inconsistencies with established architectural decisions.
        
        Args:
            new_knowledge: Extracted knowledge from new response
            project_id: Project identifier
            
        Returns:
            List of decision inconsistencies
        """
        issues = []
        
        # Get established decisions from knowledge graph
        established_decisions = self._get_established_decisions(project_id)
        
        # Check for new decisions that contradict established ones
        for decision in new_knowledge.get("decisions", []):
            decision_title = decision.get("title", "").lower()
            decision_content = decision.get("description", "")
            
            # Look for decisions about similar topics
            for est_decision in established_decisions:
                est_title = est_decision.get("title", "").lower()
                est_reasoning = est_decision.get("reasoning", "")
                
                # If decisions are about similar topics
                if self._related_decisions(decision_title, est_title):
                    # Check if reasoning contradicts
                    if self._contradicting_reasoning(
                        decision_content, 
                        est_reasoning
                    ):
                        issues.append({
                            "type": "decision_contradiction",
                            "severity": "error",
                            "decision": decision_title,
                            "established_decision": est_title,
                            "message": f"The decision '{decision_title}' contradicts established decision '{est_title}'"
                        })
        
        return issues
    
    def _get_established_decisions(self, project_id: str) -> List[Dict[str, Any]]:
        """
        Get established decisions from the knowledge graph.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of established decision dictionaries
        """
        decisions = []
        
        with self.knowledge_graph.driver.session() as session:
            result = session.run(
                """
                MATCH (d:Decision {project_id: $project_id})
                RETURN d.id as id, d.title as title, d.description as description, 
                       d.reasoning as reasoning
                """,
                project_id=project_id
            )
            
            for record in result:
                decisions.append({
                    "id": record["id"],
                    "title": record["title"],
                    "description": record["description"],
                    "reasoning": record["reasoning"]
                })
        
        return decisions
    
    def _related_decisions(self, title1: str, title2: str) -> bool:
        """
        Check if two decisions are related (about the same topic).
        
        Args:
            title1: First decision title
            title2: Second decision title
            
        Returns:
            True if decisions are related
        """
        # Extract core topics from decision titles
        # This is a simplified approach - could use NLP or vector similarity
        topics1 = set(re.findall(r'\b[a-zA-Z]{3,}\b', title1.lower()))
        topics2 = set(re.findall(r'\b[a-zA-Z]{3,}\b', title2.lower()))
        
        # Check for common topics
        common_topics = topics1 & topics2
        
        return len(common_topics) > 0
    
    def _contradicting_reasoning(self, reasoning1: str, reasoning2: str) -> bool:
        """
        Check if two reasoning statements contradict each other.
        
        Args:
            reasoning1: First reasoning statement
            reasoning2: Second reasoning statement
            
        Returns:
            True if reasoning statements contradict
        """
        # This is a simplified implementation
        # In a real system, you'd use NLP or an LLM call to detect contradictions
        
        # Look for opposite sentiment words
        positive_words = {"better", "faster", "simpler", "easier", "cleaner", "preferred", "recommended"}
        negative_words = {"worse", "slower", "complicated", "harder", "messier", "avoided", "discouraged"}
        
        # Extract words from reasoning
        words1 = set(reasoning1.lower().split())
        words2 = set(reasoning2.lower().split())
        
        # Check for contradictory sentiment
        has_positive1 = any(word in words1 for word in positive_words)
        has_negative1 = any(word in words1 for word in negative_words)
        has_positive2 = any(word in words2 for word in positive_words)
        has_negative2 = any(word in words2 for word in negative_words)
        
        # If one is positive and the other negative about the same topic
        return (has_positive1 and has_negative2) or (has_negative1 and has_positive2)
    
    def _check_structural_consistency(self, new_knowledge: Dict[str, Any], project_id: str) -> List[Dict[str, Any]]:
        """
        Check for inconsistencies in structural relationships.
        
        Args:
            new_knowledge: Extracted knowledge from new response
            project_id: Project identifier
            
        Returns:
            List of structural inconsistencies
        """
        issues = []
        
        # Get established relationships from knowledge graph
        established_relationships = self._get_established_relationships(project_id)
        
        # Check for new relationships that contradict established ones
        for rel in new_knowledge.get("relationships", []):
            from_name = rel.get("from_name", "").lower()
            to_name = rel.get("to_name", "").lower()
            rel_type = rel.get("type", "")
            
            # Skip incomplete relationships
            if not from_name or not to_name or not rel_type:
                continue
            
            # Look for contradictory relationships
            for est_rel in established_relationships:
                est_from = est_rel.get("from_name", "").lower()
                est_to = est_rel.get("to_name", "").lower()
                est_type = est_rel.get("type", "")
                
                # If same entities but different relationship type
                # (only for certain relationship types that are mutually exclusive)
                if (from_name == est_from and to_name == est_to) and rel_type != est_type:
                    if self._are_contradictory_relations(rel_type, est_type):
                        issues.append({
                            "type": "relationship_conflict",
                            "severity": "warning",
                            "entities": f"{from_name} -> {to_name}",
                            "new_relationship": rel_type,
                            "established_relationship": est_type,
                            "message": f"The relationship '{rel_type}' between {from_name} and {to_name} conflicts with established relationship '{est_type}'"
                        })
        
        return issues
    
    def _get_established_relationships(self, project_id: str) -> List[Dict[str, Any]]:
        """
        Get established relationships from the knowledge graph.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of established relationship dictionaries
        """
        relationships = []
        
        with self.knowledge_graph.driver.session() as session:
            # Query for relationships between entities
            result = session.run(
                """
                MATCH (e1:Entity {project_id: $project_id})-[r]->(e2:Entity {project_id: $project_id})
                RETURN e1.name as from_name, e2.name as to_name, type(r) as type
                """,
                project_id=project_id
            )
            
            for record in result:
                relationships.append({
                    "from_name": record["from_name"],
                    "to_name": record["to_name"],
                    "type": record["type"]
                })
        
        return relationships
    
    def _are_contradictory_relations(self, rel1: str, rel2: str) -> bool:
        """
        Check if two relationship types are contradictory.
        
        Args:
            rel1: First relationship type
            rel2: Second relationship type
            
        Returns:
            True if relationship types are contradictory
        """
        # Define sets of mutually exclusive relationship types
        mutually_exclusive = [
            {"INHERITS_FROM", "CONTAINS", "IS_PART_OF"},
            {"DEPENDS_ON", "IS_DEPENDENCY_OF"},
            {"IMPLEMENTS", "IS_IMPLEMENTED_BY"}
        ]
        
        for exclusive_set in mutually_exclusive:
            if rel1 in exclusive_set and rel2 in exclusive_set and rel1 != rel2:
                return True
        
        return False
