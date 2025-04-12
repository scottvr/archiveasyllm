"""
RESTful API routes for ArchivistLLM with OpenAPI documentation.
"""
from flask import Blueprint, request, jsonify, current_app
from flask_restx import Api, Resource, fields, Namespace
import os
import uuid
from datetime import datetime
import json
from typing import Dict, Any, List, Optional

from archivist.memory.graph import KnowledgeGraph
from archivist.memory.vector import VectorStore
from archivist.analyzer.codebase import CodebaseAnalyzer
from archivist.analyzer.consistency import ConsistencyChecker
from archivist.models.project import Project
from archivist.models.chat import Chat, Message
from archivist.llm.client import LLMClient

import config

# Create Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Initialize flask-restx API
authorizations = {
    'apikey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'X-API-Key'
    }
}

api = Api(
    api_bp,
    version='1.0',
    title='ArchivistLLM API',
    description='API for interacting with ArchivistLLM knowledge and chat functionality',
    doc='/docs',
    authorizations=authorizations,
    security='apikey'
)

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

# Create namespaces for organizing endpoints
ns_projects = api.namespace('projects', description='Project operations')
ns_chats = api.namespace('chats', description='Chat operations')
ns_knowledge = api.namespace('knowledge', description='Knowledge graph operations')
ns_codebase = api.namespace('codebase', description='Codebase analysis operations')
ns_consistency = api.namespace('consistency', description='Consistency checking operations')

# ---- Model Definitions ----

# Project models
project_base = api.model('ProjectBase', {
    'name': fields.String(required=True, description='Project name', example='My Project'),
    'description': fields.String(description='Project description', example='A sample project for testing')
})

project_create = api.model('ProjectCreate', {
    'name': fields.String(required=True, description='Project name', example='My Project'),
    'description': fields.String(description='Project description', example='A sample project for testing'),
    'codebase_path': fields.String(description='Path to codebase', example='/path/to/codebase'),
    'excluded_dirs': fields.String(description='Comma-separated list of directories to exclude', 
                                  example='node_modules,venv,.git')
})

project_update = api.model('ProjectUpdate', {
    'name': fields.String(description='Project name', example='Updated Project Name'),
    'description': fields.String(description='Project description', example='Updated description'),
    'settings': fields.Raw(description='Project settings')
})

project_full = api.model('Project', {
    'id': fields.String(readonly=True, description='Unique project identifier'),
    'name': fields.String(required=True, description='Project name'),
    'description': fields.String(description='Project description'),
    'created_at': fields.String(readonly=True, description='Creation timestamp'),
    'codebase_path': fields.String(description='Path to associated codebase'),
    'codebase_stats': fields.Raw(description='Statistics from codebase analysis')
})

# Chat models
chat_create = api.model('ChatCreate', {
    'name': fields.String(description='Chat name', example='Architecture Discussion')
})

chat_full = api.model('Chat', {
    'id': fields.String(readonly=True, description='Unique chat identifier'),
    'name': fields.String(description='Chat name'),
    'project_id': fields.String(description='Project identifier'),
    'created_at': fields.String(readonly=True, description='Creation timestamp'),
    'updated_at': fields.String(readonly=True, description='Last update timestamp'),
    'message_count': fields.Integer(description='Number of messages in chat')
})

# Message models
message_create = api.model('MessageCreate', {
    'content': fields.String(required=True, description='Message content', 
                            example='What design pattern should I use for this feature?')
})

artifact_model = api.model('Artifact', {
    'id': fields.String(description='Artifact identifier'),
    'type': fields.String(description='Artifact type (code, markdown, etc.)'),
    'language': fields.String(description='Programming language'),
    'title': fields.String(description='Artifact title'),
    'content': fields.String(description='Artifact content')
})

consistency_issue_model = api.model('ConsistencyIssue', {
    'type': fields.String(description='Issue type'),
    'severity': fields.String(description='Issue severity (warning, error)'),
    'message': fields.String(description='Issue description')
})

message_full = api.model('Message', {
    'id': fields.String(readonly=True, description='Unique message identifier'),
    'chat_id': fields.String(description='Chat identifier'),
    'content': fields.String(description='Message content'),
    'role': fields.String(description='Message role (user, assistant)'),
    'timestamp': fields.String(readonly=True, description='Creation timestamp'),
    'artifacts': fields.List(fields.Nested(artifact_model), description='Message artifacts'),
    'consistency_issues': fields.List(fields.Nested(consistency_issue_model), description='Consistency issues')
})

message_response = api.model('MessageResponse', {
    'message': fields.Nested(message_full, description='Assistant message'),
    'consistency_issues': fields.List(fields.Nested(consistency_issue_model), description='Consistency issues')
})

# Codebase models
codebase_analyze = api.model('CodebaseAnalyze', {
    'codebase_path': fields.String(description='Path to codebase', example='/path/to/codebase'),
    'excluded_dirs': fields.String(description='Comma-separated list of directories to exclude', 
                                  example='node_modules,venv,.git')
})

codebase_stats = api.model('CodebaseStats', {
    'files_processed': fields.Integer(description='Number of files processed'),
    'entities_extracted': fields.Integer(description='Number of entities extracted'),
    'relationships_extracted': fields.Integer(description='Number of relationships extracted'),
    'decisions_extracted': fields.Integer(description='Number of decisions extracted'),
    'patterns_extracted': fields.Integer(description='Number of patterns extracted'),
    'languages': fields.Raw(description='Breakdown of files by language')
})

codebase_result = api.model('CodebaseResult', {
    'success': fields.Boolean(description='Whether the analysis was successful'),
    'stats': fields.Nested(codebase_stats, description='Analysis statistics'),
    'message': fields.String(description='Result message')
})

# Knowledge models
knowledge_decision = api.model('Decision', {
    'id': fields.String(description='Decision identifier'),
    'title': fields.String(description='Decision title'),
    'description': fields.String(description='Decision description'),
    'reasoning': fields.String(description='Decision reasoning'),
    'alternatives': fields.List(fields.String, description='Alternative options')
})

knowledge_pattern = api.model('Pattern', {
    'id': fields.String(description='Pattern identifier'),
    'name': fields.String(description='Pattern name'),
    'description': fields.String(description='Pattern description'),
    'examples': fields.List(fields.String, description='Code examples')
})

knowledge_entity = api.model('Entity', {
    'id': fields.String(description='Entity identifier'),
    'name': fields.String(description='Entity name'),
    'type': fields.String(description='Entity type'),
    'file': fields.String(description='File containing the entity')
})

knowledge_relationship = api.model('Relationship', {
    'from_name': fields.String(description='Source entity name'),
    'from_type': fields.String(description='Source entity type'),
    'type': fields.String(description='Relationship type'),
    'to_name': fields.String(description='Target entity name'),
    'to_type': fields.String(description='Target entity type')
})

knowledge_search_result = api.model('SearchResult', {
    'id': fields.String(description='Result identifier'),
    'content': fields.String(description='Content snippet'),
    'source_type': fields.String(description='Source type'),
    'relevance': fields.Float(description='Relevance score')
})

# Consistency models
consistency_check = api.model('ConsistencyCheck', {
    'content': fields.String(required=True, description='Content to check'),
    'artifacts': fields.List(fields.Nested(artifact_model), description='Related artifacts'),
    'store_knowledge': fields.Boolean(description='Whether to store extracted knowledge')
})

consistency_result = api.model('ConsistencyResult', {
    'consistency_issues': fields.List(fields.Nested(consistency_issue_model), description='Consistency issues'),
    'has_issues': fields.Boolean(description='Whether any issues were found')
})

# ---- API Endpoints ----

# Helper function for API key validation
def require_api_key():
    """Check for valid API key in request."""
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        api.abort(401, 'API key is required')
    
    # In a real app, verify the API key against stored keys
    # For now, accept any non-empty key
    return True

# --- Project Endpoints ---

@ns_projects.route('/')
class ProjectList(Resource):
    @ns_projects.doc('list_projects')
    @ns_projects.marshal_list_with(project_full)
    def get(self):
        """List all projects"""
        require_api_key()
        projects = Project.list_all()
        return [project.to_dict() for project in projects]
    
    @ns_projects.doc('create_project')
    @ns_projects.expect(project_create)
    @ns_projects.marshal_with(project_full, code=201)
    def post(self):
        """Create a new project"""
        require_api_key()
        data = request.json
        if not data or 'name' not in data:
            api.abort(400, 'Project name is required')
        
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
        
        return project.to_dict(), 201

@ns_projects.route('/<string:project_id>')
@ns_projects.param('project_id', 'Project identifier')
class ProjectResource(Resource):
    @ns_projects.doc('get_project')
    @ns_projects.marshal_with(project_full)
    def get(self, project_id):
        """Get a project by ID"""
        require_api_key()
        project = Project.get(project_id)
        if not project:
            api.abort(404, 'Project not found')
        return project.to_dict()
    
    @ns_projects.doc('update_project')
    @ns_projects.expect(project_update)
    @ns_projects.marshal_with(project_full)
    def put(self, project_id):
        """Update a project"""
        require_api_key()
        project = Project.get(project_id)
        if not project:
            api.abort(404, 'Project not found')
        
        data = request.json
        if not data:
            api.abort(400, 'No data provided')
        
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
        
        return project.to_dict()
    
    @ns_projects.doc('delete_project')
    @ns_projects.response(204, 'Project deleted')
    def delete(self, project_id):
        """Delete a project"""
        require_api_key()
        project = Project.get(project_id)
        if not project:
            api.abort(404, 'Project not found')
        
        success = project.delete()
        if not success:
            api.abort(500, 'Failed to delete project')
        
        return '', 204

# --- Chat Endpoints ---

@ns_projects.route('/<string:project_id>/chats')
@ns_projects.param('project_id', 'Project identifier')
class ChatList(Resource):
    @ns_projects.doc('list_chats')
    @ns_projects.marshal_list_with(chat_full)
    def get(self, project_id):
        """List all chats for a project"""
        require_api_key()
        project = Project.get(project_id)
        if not project:
            api.abort(404, 'Project not found')
        
        chats = Chat.list_by_project(project_id)
        return [chat.to_dict() for chat in chats]
    
    @ns_projects.doc('create_chat')
    @ns_projects.expect(chat_create)
    @ns_projects.marshal_with(chat_full, code=201)
    def post(self, project_id):
        """Create a new chat in a project"""
        require_api_key()
        project = Project.get(project_id)
        if not project:
            api.abort(404, 'Project not found')
        
        data = request.json or {}
        name = data.get('name', f"Chat {Chat.count_by_project(project_id) + 1}")
        
        chat = Chat.create(project_id=project_id, name=name)
        return chat.to_dict(), 201

@ns_chats.route('/<string:chat_id>')
@ns_chats.param('chat_id', 'Chat identifier')
class ChatResource(Resource):
    @ns_chats.doc('get_chat')
    @ns_chats.marshal_with(chat_full)
    def get(self, chat_id):
        """Get a chat by ID"""
        require_api_key()
        chat = Chat.get(chat_id)
        if not chat:
            api.abort(404, 'Chat not found')
        return chat.to_dict()

@ns_chats.route('/<string:chat_id>/messages')
@ns_chats.param('chat_id', 'Chat identifier')
class MessageList(Resource):
    @ns_chats.doc('list_messages')
    @ns_chats.marshal_list_with(message_full)
    def get(self, chat_id):
        """List all messages in a chat"""
        require_api_key()
        chat = Chat.get(chat_id)
        if not chat:
            api.abort(404, 'Chat not found')
        
        messages = Message.list_by_chat(chat_id)
        return [message.to_dict() for message in messages]
    
    @ns_chats.doc('send_message')
    @ns_chats.expect(message_create)
    @ns_chats.marshal_with(message_response)
    def post(self, chat_id):
        """Send a message to the LLM and get a response"""
        require_api_key()
        chat = Chat.get(chat_id)
        if not chat:
            api.abort(404, 'Chat not found')
        
        data = request.json
        if not data or 'content' not in data:
            api.abort(400, 'Message content is required')
        
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
        
        return {
            "message": assistant_msg.to_dict(),
            "consistency_issues": consistency_issues
        }

# --- Codebase Endpoints ---

@ns_projects.route('/<string:project_id>/codebase')
@ns_projects.param('project_id', 'Project identifier')
class CodebaseAnalysis(Resource):
    @ns_projects.doc('analyze_codebase')
    @ns_projects.expect(codebase_analyze)
    @ns_projects.marshal_with(codebase_result)
    def post(self, project_id):
        """Analyze or update a codebase for a project"""
        require_api_key()
        
        # Validate project
        project = Project.get(project_id)
        if not project:
            api.abort(404, 'Project not found')
        
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
            api.abort(400, 'Invalid codebase path')
        
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
            
            return {
                "success": True, 
                "stats": stats,
                "message": f"Analyzed {stats['files_processed']} files successfully"
            }
            
        except Exception as e:
            current_app.logger.error(f"Error analyzing codebase: {e}")
            api.abort(500, f"Error analyzing codebase: {str(e)}")

# --- Knowledge Endpoints ---

@ns_projects.route('/<string:project_id>/knowledge/<string:knowledge_type>')
@ns_projects.param('project_id', 'Project identifier')
@ns_projects.param('knowledge_type', 'Type of knowledge (decisions, patterns, entities, relationships, search)')
class KnowledgeResource(Resource):
    @ns_projects.doc('get_knowledge')
    @ns_projects.param('q', 'Search query (required for search knowledge type)', _in='query')
    @ns_projects.param('limit', 'Maximum number of results', _in='query', type=int, default=1000)
    @ns_projects.param('offset', 'Number of results to skip', _in='query', type=int, default=0)
    def get(self, project_id, knowledge_type):
        """Get knowledge elements from the knowledge graph"""
        require_api_key()
        
        # Validate project
        project = Project.get(project_id)
        if not project:
            api.abort(404, 'Project not found')
        
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
                    
                    return decisions
                    
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
                    
                    return patterns
                    
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
                    
                    return entities
                    
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
                    
                    return relationships
                    
            elif knowledge_type == 'search':
                # Semantic search across all knowledge
                if not search_query:
                    api.abort(400, "Search query is required for knowledge type 'search'")
                
                # Use vector search
                results = vector_store.search(search_query, project_id, limit=limit)
                return results
                
            else:
                api.abort(400, f"Unsupported knowledge type: {knowledge_type}")
                
        except Exception as e:
            current_app.logger.error(f"Error retrieving knowledge: {e}")
            api.abort(500, f"Error retrieving knowledge: {str(e)}")

# --- Consistency Checking Endpoints ---

@ns_projects.route('/<string:project_id>/consistency')
@ns_projects.param('project_id', 'Project identifier')
class ConsistencyCheck(Resource):
    @ns_projects.doc('check_consistency')
    @ns_projects.expect(consistency_check)
    @ns_projects.marshal_with(consistency_result)
    def post(self, project_id):
        """Check text for consistency with existing knowledge"""
        require_api_key()
        
        # Validate project
        project = Project.get(project_id)
        if not project:
            api.abort(404, 'Project not found')
        
        data = request.json
        if not data or 'content' not in data:
            api.abort(400, 'Content is required')
        
        content = data['content']
        artifacts = data.get('artifacts', [])
        
        # Check consistency
        try:
            issues = consistency_checker.check(content, project_id, artifacts)
            
            # Optionally store knowledge
            if data.get('store_knowledge', False):
                _extract_and_store_knowledge(content, artifacts, project_id)
            
            return {
                "consistency_issues": issues,
                "has_issues": len(issues) > 0
            }
            
        except Exception as e:
            current_app.logger.error(f"Error checking consistency: {e}")
            api.abort(500, f"Error checking consistency: {str(e)}")

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
    from archivist.memory.extractor import KnowledgeExtractor
    
    extractor = KnowledgeExtractor()
    
    # Extract entities, relationships, decisions, etc.
    knowledge = extractor.extract(response, artifacts)
    
    # Store in knowledge graph
    knowledge_graph.store(knowledge, project_id)
    
    # Store in vector database for semantic search
    vector_store.store(response, artifacts, project_id)
    
    return knowledge