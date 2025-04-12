"""
Knowledge Graph implementation for storing architectural decisions and patterns.
"""
# crumb: memory\graph.py
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime
import logging
import neo4j
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Knowledge graph for storing and retrieving architectural decisions,
    code patterns, and reasoning.
    """
    
    def __init__(self, uri: str, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize the knowledge graph connection.
        
        Args:
            uri: Neo4j database URI
            username: Optional database username
            password: Optional database password
        """
        self.uri = uri
        auth = (username, password) if username and password else None
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=auth)
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Successfully connected to Neo4j at {uri}")
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {e}")
            raise
    
    def init_schema(self):
        """
        Initialize the database schema with constraints and indexes.
        """
        try:
            with self.driver.session() as session:
                # Create constraints for unique IDs
                constraints = [
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Pattern) REQUIRE p.id IS UNIQUE"
                ]
                
                # Create indexes for faster lookups
                indexes = [
                    "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                    "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                    "CREATE INDEX IF NOT EXISTS FOR (d:Decision) ON (d.title)",
                    "CREATE INDEX IF NOT EXISTS FOR (p:Pattern) ON (p.name)"
                ]
                
                for constraint in constraints:
                    session.run(constraint)
                
                for index in indexes:
                    session.run(index)
                
                logger.info("Neo4j schema initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Neo4j schema: {e}")
            raise
    
    def initialize_project(self, project_id: str) -> None:
        """
        Initialize a new project in the knowledge graph.
        
        Args:
            project_id: Unique project identifier
        """
        with self.driver.session() as session:
            try:
                session.run(
                    """
                    MERGE (p:Project {id: $id})
                    ON CREATE SET p.created_at = datetime()
                    """,
                    id=project_id
                )
                logger.info(f"Initialized project {project_id} in knowledge graph")
            except Exception as e:
                logger.error(f"Error initializing project {project_id}: {e}")
                raise
    
    def store(self, knowledge: Dict[str, Any], project_id: str) -> None:
        """
        Store extracted knowledge in the graph.
        
        Args:
            knowledge: Dictionary containing extracted knowledge
            project_id: Project identifier
        """
        if not knowledge:
            return
            
        entities = knowledge.get("entities", [])
        relationships = knowledge.get("relationships", [])
        decisions = knowledge.get("decisions", [])
        patterns = knowledge.get("patterns", [])
        
        logger.info(f"Storing {len(entities)} entities, {len(relationships)} relationships, "
                   f"{len(decisions)} decisions, and {len(patterns)} patterns for project {project_id}")
        
        try:
            with self.driver.session() as session:
                # Store entities (functions, classes, variables, etc.)
                for entity in entities:
                    self._store_entity(session, entity, project_id)
                
                # Store relationships between entities
                for rel in relationships:
                    self._store_relationship(session, rel, project_id)
                
                # Store architectural decisions
                for decision in decisions:
                    self._store_decision(session, decision, project_id)
                
                # Store design patterns
                for pattern in patterns:
                    self._store_pattern(session, pattern, project_id)
        except Exception as e:
            logger.error(f"Error storing knowledge: {e}")
            raise
    
    def _store_entity(self, session, entity: Dict[str, Any], project_id: str) -> None:
        """Store a code entity in the graph."""
        entity_id = entity.get("id", str(uuid.uuid4()))
        entity_type = entity.get("type", "Unknown")
        name = entity.get("name", "Unnamed")
        properties = entity.get("properties", {})
        
        # Merge to avoid duplicates
        session.run(
            """
            MATCH (p:Project {id: $project_id})
            MERGE (e:Entity {id: $id, project_id: $project_id})
            SET e.type = $type, 
                e.name = $name, 
                e.updated_at = datetime(),
                e += $properties
            MERGE (e)-[:BELONGS_TO]->(p)
            """,
            id=entity_id,
            project_id=project_id,
            type=entity_type,
            name=name,
            properties=properties
        )
    
    def _store_relationship(self, session, relationship: Dict[str, Any], project_id: str) -> None:
        """Store a relationship between entities in the graph."""
        from_id = relationship.get("from_id")
        to_id = relationship.get("to_id")
        rel_type = relationship.get("type", "RELATED_TO")
        properties = relationship.get("properties", {})
        
        if not from_id or not to_id:
            # Skip relationships without resolved entity IDs
            return
            
        # Using Cypher parameters in dynamic relationship types is challenging
        # We need to construct the query with proper escaping
        rel_type_escaped = rel_type.replace('`', '``')
        query = f"""
            MATCH (from:Entity {{id: $from_id, project_id: $project_id}})
            MATCH (to:Entity {{id: $to_id, project_id: $project_id}})
            MERGE (from)-[r:`{rel_type_escaped}`]->(to)
            SET r += $properties,
                r.updated_at = datetime()
            """
        
        session.run(
            query,
            from_id=from_id,
            to_id=to_id,
            project_id=project_id,
            properties=properties
        )
    
    def _store_decision(self, session, decision: Dict[str, Any], project_id: str) -> None:
        """Store an architectural decision in the graph."""
        decision_id = decision.get("id", str(uuid.uuid4()))
        title = decision.get("title", "Unnamed Decision")
        description = decision.get("description", "")
        reasoning = decision.get("reasoning", "")
        alternatives = decision.get("alternatives", [])
        
        session.run(
            """
            MATCH (p:Project {id: $project_id})
            MERGE (d:Decision {id: $id, project_id: $project_id})
            SET d.title = $title,
                d.description = $description,
                d.reasoning = $reasoning,
                d.alternatives = $alternatives,
                d.updated_at = datetime()
            MERGE (d)-[:BELONGS_TO]->(p)
            """,
            id=decision_id,
            project_id=project_id,
            title=title,
            description=description,
            reasoning=reasoning,
            alternatives=alternatives
        )
        
        # Link decision to affected entities
        for entity_id in decision.get("affects", []):
            session.run(
                """
                MATCH (d:Decision {id: $decision_id, project_id: $project_id})
                MATCH (e:Entity {id: $entity_id, project_id: $project_id})
                MERGE (d)-[:AFFECTS]->(e)
                """,
                decision_id=decision_id,
                entity_id=entity_id,
                project_id=project_id
            )
    
    def _store_pattern(self, session, pattern: Dict[str, Any], project_id: str) -> None:
        """Store a design pattern in the graph."""
        pattern_id = pattern.get("id", str(uuid.uuid4()))
        name = pattern.get("name", "Unnamed Pattern")
        description = pattern.get("description", "")
        examples = pattern.get("examples", [])
        
        session.run(
            """
            MATCH (p:Project {id: $project_id})
            MERGE (pat:Pattern {id: $id, project_id: $project_id})
            SET pat.name = $name,
                pat.description = $description,
                pat.examples = $examples,
                pat.updated_at = datetime()
            MERGE (pat)-[:BELONGS_TO]->(p)
            """,
            id=pattern_id,
            project_id=project_id,
            name=name,
            description=description,
            examples=examples
        )
        
        # Link pattern to involved entities
        for entity_id in pattern.get("involves", []):
            session.run(
                """
                MATCH (p:Pattern {id: $pattern_id, project_id: $project_id})
                MATCH (e:Entity {id: $entity_id, project_id: $project_id})
                MERGE (p)-[:INVOLVES]->(e)
                """,
                pattern_id=pattern_id,
                entity_id=entity_id,
                project_id=project_id
            )
    
    def get_relevant_context(self, prompt: str, project_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant context from the knowledge graph based on a prompt.
        
        This is a simplified implementation. In a real system, you'd use
        more sophisticated relevance scoring.
        
        Args:
            prompt: The user prompt
            project_id: Project identifier
            limit: Maximum number of context items to return
            
        Returns:
            List of relevant context items
        """
        # Extract keywords for matching
        keywords = self._extract_keywords(prompt)
        
        if not keywords:
            return []
            
        result = []
        
        try:
            with self.driver.session() as session:
                # Get relevant decisions
                decisions = session.run(
                    """
                    MATCH (d:Decision {project_id: $project_id})
                    WHERE any(keyword IN $keywords WHERE 
                        d.title CONTAINS keyword OR 
                        d.description CONTAINS keyword OR
                        d.reasoning CONTAINS keyword)
                    RETURN d.id as id, d.title as title, d.description as description, 
                           d.reasoning as reasoning, 'decision' as type
                    ORDER BY d.updated_at DESC
                    LIMIT $limit
                    """,
                    project_id=project_id,
                    keywords=keywords,
                    limit=limit
                )
                
                for decision in decisions:
                    result.append({
                        "id": decision["id"],
                        "title": decision["title"],
                        "content": f"{decision['description']} Reasoning: {decision['reasoning']}",
                        "type": "decision"
                    })
                
                # Get relevant patterns
                patterns = session.run(
                    """
                    MATCH (p:Pattern {project_id: $project_id})
                    WHERE any(keyword IN $keywords WHERE 
                        p.name CONTAINS keyword OR 
                        p.description CONTAINS keyword)
                    RETURN p.id as id, p.name as name, p.description as description, 
                           'pattern' as type
                    ORDER BY p.updated_at DESC
                    LIMIT $limit
                    """,
                    project_id=project_id,
                    keywords=keywords,
                    limit=limit
                )
                
                for pattern in patterns:
                    result.append({
                        "id": pattern["id"],
                        "title": pattern["name"],
                        "content": pattern["description"],
                        "type": "pattern"
                    })
                
                # Get relevant entities
                entities = session.run(
                    """
                    MATCH (e:Entity {project_id: $project_id})
                    WHERE any(keyword IN $keywords WHERE 
                        e.name CONTAINS keyword)
                    RETURN e.id as id, e.name as name, e.type as type
                    ORDER BY e.type, e.name
                    LIMIT $limit
                    """,
                    project_id=project_id,
                    keywords=keywords,
                    limit=limit
                )
                
                for entity in entities:
                    result.append({
                        "id": entity["id"],
                        "title": entity["name"],
                        "content": f"{entity['type']}: {entity['name']}",
                        "type": "entity"
                    })
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            # Return empty result on error rather than crashing
        
        return result
    
    def _extract_keywords(self, prompt: str) -> List[str]:
        """
        Extract keywords from a prompt.
        
        This is a very simplified implementation. In a real system,
        you'd use NLP techniques, possibly even an LLM call, to
        extract relevant keywords.
        
        Args:
            prompt: The user prompt
            
        Returns:
            List of keywords
        """
        # Remove common words and punctuation
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "like"}
        words = prompt.lower().split()
        keywords = [word.strip(".,?!()[]{}:;\"'") for word in words if len(word) > 3 and word not in common_words]
        
        # Remove duplicates
        return list(set(keywords))
    
    def clear_project(self, project_id: str) -> None:
        """
        Clear all knowledge for a project.
        
        Args:
            project_id: Project identifier
        """
        with self.driver.session() as session:
            session.run(
                """
                MATCH (n)-[r]->(m)
                WHERE n.project_id = $project_id OR m.project_id = $project_id
                DELETE r
                """,
                project_id=project_id
            )
            
            session.run(
                """
                MATCH (n)
                WHERE n.project_id = $project_id
                DETACH DELETE n
                """,
                project_id=project_id
            )
    
    def get_project_stats(self, project_id: str) -> Dict[str, int]:
        """
        Get statistics about the knowledge graph for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Dictionary with statistics
        """
        with self.driver.session() as session:
            # Count entities by type
            entities_result = session.run(
                """
                MATCH (e:Entity {project_id: $project_id})
                RETURN e.type as type, count(e) as count
                """,
                project_id=project_id
            )
            
            entities_by_type = {record["type"]: record["count"] for record in entities_result}
            
            # Count decisions
            decisions_count = session.run(
                """
                MATCH (d:Decision {project_id: $project_id})
                RETURN count(d) as count
                """,
                project_id=project_id
            ).single()["count"]
            
            # Count patterns
            patterns_count = session.run(
                """
                MATCH (p:Pattern {project_id: $project_id})
                RETURN count(p) as count
                """,
                project_id=project_id
            ).single()["count"]
            
            # Count relationships
            relationships_result = session.run(
                """
                MATCH (e1:Entity {project_id: $project_id})-[r]->(e2:Entity {project_id: $project_id})
                RETURN type(r) as type, count(r) as count
                """,
                project_id=project_id
            )
            
            relationships_by_type = {record["type"]: record["count"] for record in relationships_result}
            
            return {
                "entities_total": sum(entities_by_type.values()),
                "entities_by_type": entities_by_type,
                "decisions": decisions_count,
                "patterns": patterns_count,
                "relationships_total": sum(relationships_by_type.values()),
                "relationships_by_type": relationships_by_type
            }
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
