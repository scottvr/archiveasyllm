"""
RESTful API routes for ArchiveAsyLLM.
"""
from flask import Blueprint, request, jsonify, current_app
from flask_restx import Api, Resource, fields, Namespace
import os
import uuid
from datetime import datetime
import json

from archiveasy.memory.graph import KnowledgeGraph
from archiveasy.memory.vector import VectorStore
from archiveasy.analyzer.codebase import CodebaseAnalyzer
from archiveasy.analyzer.consistency import ConsistencyChecker
from archiveasy.models.project import Project
from archiveasy.models.chat import Chat, Message
from archiveasy.llm.client import LLMClient

import config

# Create Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Initialize components
knowledge_graph = None
vector_store = None
llm_client = None
consistency_checker = None

def init_api(app):
    """Initialize API components with app context."""
    global knowledge_graph, vector_store, llm_client, consistency_checker
    
    knowledge_graph = app.config.get('knowledge_graph')
    vector_store = app.config.get('vector_store')
    llm_client = app.config.get('llm_client')
    consistency_checker = app.config.get('consistency_checker')

# Authentication middleware (placeholder - implement proper auth)
def require_api_key():
    """Check for valid API key in request."""
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        return False, {'error': 'API key is required'}, 401
    
    # In a real app, verify the API key against stored keys
    # For now, accept any non-empty key
    return True, None, None

# Error handler
@api_bp.errorhandler(Exception)
def handle_error(e):
    """Handle API errors."""
    current_app.logger.error(f"API error: {str(e)}")
    return jsonify({'error': str(e)}), 500

# API Documentation
@api_bp.route('/', methods=['GET'])
def api_docs():
    """Return API documentation."""
    return jsonify({
        'name': 'ArchiveAsyLLM API',
        'version': 'v1',
        'description': 'API for interacting with ArchiveAsyLLM knowledge and chat functionality',
        'endpoints': {
            'projects': '/api/v1/projects',
            'project': '/api/v1/projects/{project_id}',
            'chats': '/api/v1/projects/{project_id}/chats',
            'messages': '/api/v1/chats/{chat_id}/messages',
            'codebase': '/api/v1/projects/{project_id}/codebase',
            'knowledge': '/api/v1/projects/{project_id}/knowledge/{knowledge_type}',
            'consistency': '/api/v1/projects/{project_id}/consistency'
        }
    })

# Project endpoints
@api_bp.route('/projects', methods=['GET'])
def list_projects():
    """List all projects."""
    auth_ok, error_response, error_code = require_api_key()
    if not auth_ok:
        return jsonify(error_response), error_code
    
    projects = Project.list_all()
    return jsonify([project.to_dict() for project in projects])

@api_bp.route('/projects', methods=['POST'])
def create_project():
    """Create a new project."""
    auth_ok, error_response, error_code = require_api_key()
    if not auth_ok:
        return jsonify(error_response), error_code
    
    data = request.json
    if not data or 'name' not in data:
        return jsonify({'error': 'Project name is required'}), 400
    
    name = data['name']
    description = data.get('description', '')
    
    project = Project.create(name=name, description=description)
    
    # Initialize knowledge graph for this project
    knowledge_graph.initialize_project(project.id)
    
    # Initialize codebase if path provided
    codebase_path = data.get('codebase_path')
    if codebase_path and os.path.isdir(codebase_path):
        try:
            analyzer = CodebaseAnalyzer(knowledge_graph, vector_store)
            excluded_dirs = data.get('excluded_dirs', 'node_modules,venv,.venv,env,__pycache__,.git,dist,build')
            
            stats = analyzer.analyze_codebase(
                codebase_path, 
                project.id, 
                excluded_dirs=excluded_dirs.split(',') if isinstance(excluded_dirs, str) else excluded_dirs
            )
            
            # Update project with codebase info
            project.update(
                codebase_path=os.path.abspath(codebase_path),
                codebase_stats=stats
            )
            
            # Save in project config
            project_config = config.load_project_config(project.id)
            if not project_config:
                project_config = {}
            
            if "codebase" not in project_config:
                project_config["codebase"] = {}
                
            project_config["codebase"].update({
                "path": os.path.abspath(codebase_path),
                "last_analyzed": datetime.now().isoformat(),
                "stats": stats
            })
            
            config.save_project_config(project.id, project_config)
            
        except Exception as e:
            current_app.logger.error(f"Error analyzing codebase: {e}")
            # Continue without failing the project creation
    
    return jsonify(project.to_dict()), 201

@api_bp.route('/projects/<project_id>', methods=['GET'])
def get_project(project_id):
    """Get project details."""
    auth_ok, error_response, error_code = require_api_key()
    if not auth_ok:
        return jsonify(error_response), error_code
    
    project = Project.get(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    
    return jsonify(project.to_dict())

@api_bp.route('/projects/<project_id>', methods=['PUT'])
def update_project(project_id):
    """Update project details."""
    auth_ok, error_response, error_code = require_api_key()
    if not auth_ok:
        return jsonify(error_response), error_code
    
    project = Project.get(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Update basic project info
    for field in ['name', 'description']:
        if field in data:
            project.update(**{field: data[field]})
    
    # Update settings if provided
    if 'settings' in data:
        project_config = config.load_project_config(project_id)
        if not project_config:
            project_config = {}
        
        # Update memory settings
        if 'memory' in data['settings']:
            if 'memory' not in project_config:
                project_config['memory'] = {}
                
            project_config['memory'].update(data['settings']['memory'])
        
        # Update LLM settings
        if 'llm' in data['settings']:
            if 'llm' not in project_config:
                project_config['llm'] = {}
                
            project_config['llm'].update(data['settings']['llm'])
        
        # Save updated config
        config.save_project_config(project_id, project_config)
    
    return jsonify(project.to_dict())

@api_bp.route('/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Delete a project."""
    auth_ok, error_response, error_code = require_api_key()
    if not auth_ok:
        return jsonify(error_response), error_code
    
    project = Project.get(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    
    success = project.delete()
    if not success:
        return jsonify({'error': 'Failed to delete project'}), 500
    
    return jsonify({'success': True})

# Chat endpoints
@api_bp.route('/projects/<project_id>/chats', methods=['GET'])
def list_chats(project_id):
    """List all chats for a project."""
    auth_ok, error_response, error_code = require_api_key()
    if not auth_ok:
        return jsonify(error_response), error_code
    
    project = Project.get(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    
    chats = Chat.list_by_project(project_id)
    return jsonify([chat.to_dict() for chat in chats])

@api_bp.route('/projects/<project_id>/chats', methods=['POST'])
def create_chat(project_id):
    """Create a new chat in a project."""
    auth_ok, error_response, error_code = require_api_key()
    if not auth_ok:
        return jsonify(error_response), error_code
    
    project = Project.get(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    
    data = request.json or {}
    name = data.get('name', f"Chat {Chat.count_by_project(project_id) + 1}")
    
    chat = Chat.create(project_id=project_id, name=name)
    return jsonify(chat.to_dict()), 201

@api_bp.route('/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    """Get chat details."""
    auth_ok, error_response, error_code = require_api_key()
    if not auth_ok:
        return jsonify(error_response), error_code
    
    chat = Chat.get(chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    
    return jsonify(chat.to_dict())

@api_bp.route('/chats/<chat_id>/messages', methods=['GET'])
def list_messages(chat_id):
    """List all messages in a chat."""
    auth_ok, error_response, error_code = require_api_key()
    if not auth_ok:
        return jsonify(error_response), error_code
    
    chat = Chat.get(chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    
    messages = Message.list_by_chat(chat_id)
    return jsonify([message.to_dict() for message in messages])

@api_bp.route('/chats/<chat_id>/messages', methods=['POST'])
def send_message(chat_id):
    """Send a message to the LLM and get a response."""
    auth_ok, error_response, error_code = require_api_key()
    if not auth_ok:
        return jsonify(error_response), error_code
    
    chat = Chat.get(chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    
    data = request.json
    if not data or 'content' not in data:
        return jsonify({'error': 'Message content is required'}), 400
    
    prompt = data['content']
    
    # Store user message
    user_msg = Message.create(
        chat_id=chat_id, 
        content=prompt, 
        role="user"
    )
    
    # Get project config for settings
    project_id = chat.project_id
    project_config = config.get_merged_config(project_id, chat_id)
    
    # Check if we should use project context
    use_project_context = project_config.get('memory', {}).get('use_project_context', True)
    
    # Get relevant context if enabled
    context = None
    if use_project_context:
        context = _build_context(prompt, project_id)
    
    # Send to LLM with enhanced context
    response, artifacts = llm_client.generate(prompt, context)
    
    # Check for consistency with existing knowledge
    consistency_issues = []
    if project_config.get('memory', {}).get('consistency_checking', True):
        consistency_issues = consistency_checker.check(response, project_id, artifacts)
    
    # Extract and store new knowledge
    if project_config.get('memory', {}).get('extract_patterns', True) or \
       project_config.get('memory', {}).get('extract_decisions', True) or \
       project_config.get('memory', {}).get('extract_entities', True):
        _extract_and_store_knowledge(response, artifacts, project_id)
    
    # Store assistant response
    assistant_msg = Message.create(
        chat_id=chat_id, 
        content=response, 
        role="assistant",
        artifacts=artifacts,
        consistency_issues=consistency_issues
    )
    
    return jsonify({
        "message": assistant_msg.to_dict(),
        "consistency_issues": consistency_issues
    })

# Codebase endpoints
@api_bp.route('/projects/<project_id>/codebase', methods=['POST'])
def analyze_codebase(project_id):
    """Analyze or update a codebase for a project."""
    auth_ok, error_response, error_code = require_api_key()
    if not auth_ok:
        return jsonify(error_response), error_code
    
    # Validate project
    project = Project.get(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    
    data = request.json or {}
    codebase_path = data.get('codebase_path')
    excluded_dirs = data.get('excluded_dirs', 'node_modules,venv,.venv,env,__pycache__,.git,dist,build')
    
    # Use existing path if not provided
    if not codebase_path:
        project_config = config.load_project_config(project_id)
        if project_config and "codebase" in project_config and "path" in project_config["codebase"]:
            codebase_path = project_config["codebase"]["path"]
    
    # Validate codebase path
    if not codebase_path or not os.path.isdir(codebase_path):
        return jsonify({"error": "Invalid codebase path"}), 400
    
    try:
        # Initialize analyzer
        analyzer = CodebaseAnalyzer(knowledge_graph, vector_store)
        
        # Analyze codebase
        stats = analyzer.analyze_codebase(
            codebase_path, 
            project_id, 
            excluded_dirs=excluded_dirs.split(',') if isinstance(excluded_dirs, str) else excluded_dirs
        )
        
        # Update project with codebase info
        project.update(
            codebase_path=os.path.abspath(codebase_path),
            codebase_stats=stats
        )
        
        # Save in project config
        project_config = config.load_project_config(project_id)
        if not project_config:
            project_config = {}
        
        if "codebase" not in project_config:
            project_config["codebase"] = {}
            
        project_config["codebase"].update({
            "path": os.path.abspath(codebase_path),
            "last_analyzed": datetime.now().isoformat(),
            "stats": stats
        })
        
        config.save_project_config(project_id, project_config)
        
        return jsonify({
            "success": True, 
            "stats": stats,
            "message": f"Analyzed {stats['files_processed']} files successfully"
        })
        
    except Exception as e:
        current_app.logger.error(f"Error analyzing codebase: {e}")
        return jsonify({"error": str(e)}), 500

# Knowledge endpoints
@api_bp.route('/projects/<project_id>/knowledge/<knowledge_type>', methods=['GET'])
def get_knowledge(project_id, knowledge_type):
    """
    Get knowledge elements from the knowledge graph.
    
    Args:
        project_id: Project identifier
        knowledge_type: Type of knowledge to retrieve (decisions, patterns, entities, relationships)
    """
    auth_ok, error_response, error_code = require_api_key()
    if not auth_ok:
        return jsonify(error_response), error_code
    
    # Validate project
    project = Project.get(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    
    # Optional filtering parameters
    search_query = request.args.get('q', '')
    limit = min(int(request.args.get('limit', 1000)), 1000)  # Max 1000 items
    offset = int(request.args.get('offset', 0))
    
    try:
        # Query knowledge graph based on type
        if knowledge_type == 'decisions':
            # Get architectural decisions
            with knowledge_graph.driver.session() as session:
                query = """
                MATCH (d:Decision {project_id: $project_id})
                """
                
                if search_query:
                    query += """
                    WHERE d.title CONTAINS $search OR d.description CONTAINS $search
                    """
                
                query += """
                RETURN d.id as id, d.title as title, d.description as description, 
                       d.reasoning as reasoning, d.alternatives as alternatives
                ORDER BY d.title
                SKIP $offset
                LIMIT $limit
                """
                
                result = session.run(
                    query,
                    project_id=project_id,
                    search=search_query,
                    offset=offset,
                    limit=limit
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
                query = """
                MATCH (p:Pattern {project_id: $project_id})
                """
                
                if search_query:
                    query += """
                    WHERE p.name CONTAINS $search OR p.description CONTAINS $search
                    """
                
                query += """
                RETURN p.id as id, p.name as name, p.description as description, 
                       p.examples as examples
                ORDER BY p.name
                SKIP $offset
                LIMIT $limit
                """
                
                result = session.run(
                    query,
                    project_id=project_id,
                    search=search_query,
                    offset=offset,
                    limit=limit
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
                query = """
                MATCH (e:Entity {project_id: $project_id})
                WHERE e.type <> 'file'
                """
                
                if search_query:
                    query += """
                    AND (e.name CONTAINS $search OR e.type CONTAINS $search)
                    """
                
                query += """
                OPTIONAL MATCH (e)-[:DEFINED_IN]->(f:Entity {type: 'file'})
                RETURN e.id as id, e.name as name, e.type as type, 
                       f.name as file
                ORDER BY e.type, e.name
                SKIP $offset
                LIMIT $limit
                """
                
                result = session.run(
                    query,
                    project_id=project_id,
                    search=search_query,
                    offset=offset,
                    limit=limit
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
                query = """
                MATCH (e1:Entity {project_id: $project_id})-[r]->(e2:Entity {project_id: $project_id})
                WHERE e1.type <> 'file' AND e2.type <> 'file'
                """
                
                if search_query:
                    query += """
                    AND (e1.name CONTAINS $search OR e2.name CONTAINS $search OR type(r) CONTAINS $search)
                    """
                
                query += """
                RETURN e1.name as from_name, e1.type as from_type,
                       type(r) as type, e2.name as to_name, 
                       e2.type as to_type
                ORDER BY type(r), e1.name, e2.name
                SKIP $offset
                LIMIT $limit
                """
                
                result = session.run(
                    query,
                    project_id=project_id,
                    search=search_query,
                    offset=offset,
                    limit=limit
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
                
        elif knowledge_type == 'search':
            # Semantic search across all knowledge
            if not search_query:
                return jsonify({"error": "Search query is required for knowledge type 'search'"}), 400
            
            # Use vector search
            results = vector_store.search(search_query, project_id, limit=limit)
            return jsonify(results)
            
        else:
            return jsonify({"error": f"Unsupported knowledge type: {knowledge_type}"}), 400
            
    except Exception as e:
        current_app.logger.error(f"Error retrieving knowledge: {e}")
        return jsonify({"error": str(e)}), 500

# Consistency checking endpoint
@api_bp.route('/projects/<project_id>/consistency', methods=['POST'])
def check_consistency(project_id):
    """
    Check text for consistency with existing knowledge.
    
    This endpoint allows external tools to leverage the consistency checker
    without going through the chat flow.
    """
    auth_ok, error_response, error_code = require_api_key()
    if not auth_ok:
        return jsonify(error_response), error_code
    
    # Validate project
    project = Project.get(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    
    data = request.json
    if not data or 'content' not in data:
        return jsonify({"error": "Content is required"}), 400
    
    content = data['content']
    artifacts = data.get('artifacts', [])
    
    # Check consistency
    try:
        issues = consistency_checker.check(content, project_id, artifacts)
        
        # Optionally store knowledge
        if data.get('store_knowledge', False):
            _extract_and_store_knowledge(content, artifacts, project_id)
        
        return jsonify({
            "consistency_issues": issues,
            "has_issues": len(issues) > 0
        })
        
    except Exception as e:
        current_app.logger.error(f"Error checking consistency: {e}")
        return jsonify({"error": str(e)}), 500

# Helper functions
def _build_context(prompt, project_id):
    """Build enhanced context from knowledge graph and vector store."""
    # Get relevant nodes from knowledge graph
    graph_context = knowledge_graph.get_relevant_context(prompt, project_id)
    
    # Get relevant vectorized content
    vector_context = vector_store.search(prompt, project_id)
    
    # Combine contexts
    return {
        "graph_context": graph_context,
        "vector_context": vector_context
    }

def _extract_and_store_knowledge(response, artifacts, project_id):
    """Extract knowledge from response and store in knowledge systems."""
    from archiveasy.memory.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor()
    
    # Extract entities, relationships, decisions, etc.
    knowledge = extractor.extract(response, artifacts)
    
    # Store in knowledge graph
    knowledge_graph.store(knowledge, project_id)
    
    # Store in vector database for semantic search
    vector_store.store(response, artifacts, project_id)
    
    return knowledge
