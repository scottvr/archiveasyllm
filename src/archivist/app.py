@app.route('/api/project/<project_id>/knowledge/<knowledge_type>', methods=['GET'])
def get_knowledge(project_id, knowledge_type):
    """
    Get knowledge elements from the knowledge graph.
    
    Args:
        project_id: Project identifier
        knowledge_type: Type of knowledge to retrieve (decisions, patterns, entities, relationships)
    """
    # Validate project
    project = Project.get(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    
    try:
        # Query knowledge graph based on type
        if knowledge_type == 'decisions':
            # Get architectural decisions
            with knowledge_graph.driver.session() as session:
                result = session.run(
                    """
                    MATCH (d:Decision {project_id: $project_id})
                    RETURN d.id as id, d.title as title, d.description as description, 
                           d.reasoning as reasoning, d.alternatives as alternatives
                    ORDER BY d.title
                    """,
                    project_id=project_id
                )
                
                decisions = []
                for record in result:
                    decisions.append({
                        "id": record["id"],
                        "title": record["title"],
                        "description": record["description"],
                        "reasoning": record["reasoning"],
                        "alternatives": record["alternatives"] if record["alternatives"] else []
                    })
                
                return jsonify(decisions)
                
        elif knowledge_type == 'patterns':
            # Get design patterns
            with knowledge_graph.driver.session() as session:
                result = session.run(
                    """
                    MATCH (p:Pattern {project_id: $project_id})
                    RETURN p.id as id, p.name as name, p.description as description, 
                           p.examples as examples
                    ORDER BY p.name
                    """,
                    project_id=project_id
                )
                
                patterns = []
                for record in result:
                    patterns.append({
                        "id": record["id"],
                        "name": record["name"],
                        "description": record["description"],
                        "examples": record["examples"] if record["examples"] else []
                    })
                
                return jsonify(patterns)
                
        elif knowledge_type == 'entities':
            # Get code entities
            with knowledge_graph.driver.session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity {project_id: $project_id})
                    WHERE e.type <> 'file'
                    OPTIONAL MATCH (e)-[:DEFINED_IN]->(f:Entity {type: 'file'})
                    RETURN e.id as id, e.name as name, e.type as type, 
                           f.name as file
                    ORDER BY e.type, e.name
                    LIMIT 1000
                    """,
                    project_id=project_id
                )
                
                entities = []
                for record in result:
                    entities.append({
                        "id": record["id"],
                        "name": record["name"],
                        "type": record["type"],
                        "file": record["file"]
                    })
                
                return jsonify(entities)
                
        elif knowledge_type == 'relationships':
            # Get relationships between entities
            with knowledge_graph.driver.session() as session:
                result = session.run(
                    """
                    MATCH (e1:Entity {project_id: $project_id})-[r]->(e2:Entity {project_id: $project_id})
                    WHERE e1.type <> 'file' AND e2.type <> 'file'
                    RETURN e1.name as from_name, e1.type as from_type,
                           type(r) as type, e2.name as to_name, 
                           e2.type as to_type
                    ORDER BY type(r), e1.name, e2.name
                    LIMIT 1000
                    """,
                    project_id=project_id
                )
                
                relationships = []
                for record in result:
                    relationships.append({
                        "from_name": record["from_name"],
                        "from_type": record["from_type"],
                        "type": record["type"],
                        "to_name": record["to_name"],
                        "to_type": record["to_type"]
                    })
                
                return jsonify(relationships)
        else:
            return jsonify({"error": f"Unsupported knowledge type: {knowledge_type}"}), 400
            
    except Exception as e:
        app.logger.error(f"Error retrieving knowledge: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/project/<project_id>/settings', methods=['POST'])
def update_project_settings(project_id):
    """Update project settings."""
    # Validate project
    project = Project.get(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    
    data = request.json
    
    # Update project name and description
    if 'name' in data:
        project.update(name=data['name'])
    
    if 'description' in data:
        project.update(description=data['description'])
    
    # Update project settings
    if 'settings' in data:
        project_config = config.load_project_config(project_id)
        
        # Update memory settings
        if 'memory' in data['settings']:
            if not project_config:
                project_config = {}
            
            if 'memory' not in project_config:
                project_config['memory'] = {}
                
            project_config['memory'].update(data['settings']['memory'])
            
            # Save updated config
            config.save_project_config(project_id, project_config)
    
    return jsonify({"success": True})
