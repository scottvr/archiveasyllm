#!/usr/bin/env python3
"""
ArchiveAsyLLM - A framework for maintaining LLM reasoning and consistency.
"""
import os
from flask import Flask, render_template, request, jsonify, session
from archiveasy.llm.client import LLMClient
from archiveasy.memory.graph import KnowledgeGraph
from archiveasy.memory.vector import VectorStore
from archiveasy.analyzer.consistency import ConsistencyChecker
from archiveasy.models.chat import Chat, Message
from archiveasy.models.project import Project
from archiveasy.api.routes import api_bp, init_api

# Import security components
from archiveasy.security.validator import create_validator
from archiveasy.security.scanner import scan_artifacts, scan_message

import config

app = Flask(__name__, 
           template_folder='archiveasy/templates',
           static_folder='archiveasy/static')

app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# Initialize components
llm_client = LLMClient.from_config(config.llm_config)
knowledge_graph = KnowledgeGraph(config.graph_db_url, config.graph_db_user, config.graph_db_password)
vector_store = VectorStore(config.vector_db_config)
consistency_checker = ConsistencyChecker(knowledge_graph, vector_store)

# Make components available to API
app.config['llm_client'] = llm_client
app.config['knowledge_graph'] = knowledge_graph
app.config['vector_store'] = vector_store
app.config['consistency_checker'] = consistency_checker

# Initialize security components
package_validator = create_validator(
    mode=config.security_config["package_validation"]["mode"],
    whitelist_path=config.security_config["package_validation"]["whitelist_path"]
)
app.config['package_validator'] = package_validator

# Register API blueprint
app.register_blueprint(api_bp)
init_api(app)

# Ensure CORS settings allow access to documentation
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-API-Key')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def index():
    """Render the main project listing page."""
    projects = Project.list_all()
    return render_template('projects.html', projects=projects)

@app.route('/project/new', methods=['POST'])
def new_project():
    """Create a new project."""
    name = request.form.get('name')
    description = request.form.get('description', '')
    codebase_path = request.form.get('codebase_path', '')
    
    project = Project.create(name=name, description=description)
    
    # Initialize knowledge graph for this project
    knowledge_graph.initialize_project(project.id)
    
    # If codebase path is provided, analyze it
    if codebase_path and os.path.isdir(codebase_path):
        try:
            from archiveasy.analyzer.codebase import CodebaseAnalyzer
            
            # Initialize analyzer
            analyzer = CodebaseAnalyzer(knowledge_graph, vector_store)
            
            # Analyze codebase
            excluded_dirs = request.form.get('excluded_dirs', 'node_modules,venv,.venv,env,__pycache__,.git,dist,build')
            stats = analyzer.analyze_codebase(
                codebase_path, 
                project.id, 
                excluded_dirs=excluded_dirs.split(',')
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
            app.logger.error(f"Error analyzing codebase: {e}")
            # Continue without failing the project creation
    
    return jsonify({"id": project.id, "name": project.name})

@app.route('/project/<project_id>')
def view_project(project_id):
    """View a specific project and its chats."""
    project = Project.get(project_id)
    chats = Chat.list_by_project(project_id)
    return render_template('project.html', project=project, chats=chats)

@app.route('/project/<project_id>/delete', methods=['POST'])
def delete_project(project_id):
    """Delete a project."""
    # Delete vector index
    vector_store.delete_project(project_id)

    # Delete graph data
    knowledge_graph.delete_project_nodes(project_id)

    # Delete all chat data
    for chat in get_chats_for_project(project_id):
        delete_chat_files(chat["id"])
        chat_store.remove(chat["id"])

    # Delete project metadata
    project_store.remove(project_id)

    flash("Project deleted successfully.", "success")
    return redirect(url_for('index'))

@app.route('/chat/new', methods=['POST'])
def new_chat():
    """Create a new chat in a project."""
    project_id = request.form.get('project_id')
    name = request.form.get('name', f"Chat {Chat.count_by_project(project_id) + 1}")
    chat = Chat.create(project_id=project_id, name=name)
    return jsonify({"id": chat.id, "name": chat.name})

@app.route('/chat/<chat_id>')
def view_chat(chat_id):
    """View a specific chat."""
    chat = Chat.get(chat_id)
    messages = Message.list_by_chat(chat_id)
    project = Project.get(chat.project_id)
    
    # Set current chat in session
    session['current_chat_id'] = chat_id
    session['current_project_id'] = chat.project_id
    
    return render_template('chat.html', chat=chat, messages=messages, project=project)

@app.route('/chat/<chat_id>/delete', methods=['POST'])
def delete_chat(chat_id):
    """Delete a chat."""
    # Get the project ID for redirect
    chat = get_chat(chat_id)
    project_id = chat["project_id"]

    # Delete from vector store
    vector_store.delete_chat(chat_id)

    # Delete from graph
    knowledge_graph.delete_chat_nodes(chat_id)

    # Delete chat metadata/artifacts from disk
    delete_chat_files(chat_id)

    # Remove from whatever in-memory or persistent store you're using
    chat_store.remove(chat_id)

    flash("Chat deleted successfully.", "success")
    return redirect(url_for('project_view', project_id=project_id))

@app.route('/api/message', methods=['POST'])
def send_message():
    """Send a message to the LLM and get a response."""
    prompt = request.json.get('prompt')
    chat_id = request.json.get('chat_id', session.get('current_chat_id'))
    project_id = request.json.get('project_id', session.get('current_project_id'))
    
    if not chat_id or not prompt:
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Store user message
    user_msg = Message.create(
        chat_id=chat_id, 
        content=prompt, 
        role="user"
    )
    
    # Get relevant context from knowledge graph and vector store
    context = _build_context(prompt, project_id)
    
    # Send to LLM with enhanced context
    response, artifacts = llm_client.generate(prompt, context)
    
    # Check for consistency with existing knowledge
    consistency_issues = consistency_checker.check(response, project_id)
    
    # NEW: Scan response for security issues
    project_config = config.get_merged_config(project_id, chat_id)
    security_config = project_config.get('security', {})
    
    if security_config.get('scan_generated_code', True):
        # Scan message text
        message_issues = scan_message(
            response, 
            mode=security_config.get('package_validation', 'verify')
        )
        
        # Scan artifacts
        artifact_issues = scan_artifacts(
            artifacts, 
            mode=security_config.get('package_validation', 'verify')
        )
        
        # Combine security issues with consistency issues
        for issue in message_issues + artifact_issues:
            consistency_issues.append({
                "type": "security_issue",
                "severity": issue.get("severity", "warning"),
                "message": f"Security issue in generated code: {issue.get('message', '')} - {issue.get('package', '')} at line {issue.get('line', 'unknown')}"
            })
    
    # Extract and store new knowledge
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

@app.route('/api/project/<project_id>/analyze-codebase', methods=['POST'])
def analyze_codebase(project_id):
    """Analyze or update a codebase for a project."""
    codebase_path = request.json.get('codebase_path')
    excluded_dirs = request.json.get('excluded_dirs', 'node_modules,venv,.venv,env,__pycache__,.git,dist,build')
    
    # Validate project
    project = Project.get(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    
    # Use existing path if not provided
    if not codebase_path:
        project_config = config.load_project_config(project_id)
        if project_config and "codebase" in project_config and "path" in project_config["codebase"]:
            codebase_path = project_config["codebase"]["path"]
    
    # Validate codebase path
    if not codebase_path or not os.path.isdir(codebase_path):
        return jsonify({"error": "Invalid codebase path"}), 400
    
    try:
        from archiveasy.analyzer.codebase import CodebaseAnalyzer
        
        # Initialize analyzer
        analyzer = CodebaseAnalyzer(knowledge_graph, vector_store)
        
        # Analyze codebase
        stats = analyzer.analyze_codebase(
            codebase_path, 
            project_id, 
            excluded_dirs=excluded_dirs.split(',')
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
        
        # NEW: Analyze codebase dependencies for security issues
        try:
            from archiveasy.security.analyzer import analyze_project_dependencies
            
            security_results = analyze_project_dependencies(
                project_path=codebase_path,
                mode=config.security_config["package_validation"]["mode"],
                whitelist_path=config.security_config["package_validation"]["whitelist_path"],
                save_report=True
            )
            
            # Add security results to stats
            stats["security"] = {
                "packages_analyzed": len(security_results["packages"]["imports"]) + len(security_results["packages"]["requirements"]),
                "issues_found": len(security_results["packages"]["invalid"]),
                "issues": security_results["packages"]["invalid"]
            }
            
            # Update config with security results
            project_config["codebase"]["security"] = stats["security"]
            config.save_project_config(project_id, project_config)
            
        except Exception as security_e:
            app.logger.error(f"Error analyzing codebase dependencies: {security_e}")
            # Don't fail the overall process if security scanning fails
        
        return jsonify({
            "success": True, 
            "stats": stats,
            "message": f"Analyzed {stats['files_processed']} files successfully"
        })
        
    except Exception as e:
        app.logger.error(f"Error analyzing codebase: {e}")
        return jsonify({"error": str(e)}), 500

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
                
        # NEW: Add a security endpoint for package motivations
        elif knowledge_type == 'dependencies':
            # Get project codebase path
            project_config = config.load_project_config(project_id)
            codebase_path = project_config.get("codebase", {}).get("path", "")
            
            if not codebase_path or not os.path.isdir(codebase_path):
                return jsonify({"error": "No valid codebase path found for project"}), 400
            
            # Run dependency analysis
            from archiveasy.security.analyzer import analyze_project_dependencies
            
            results = analyze_project_dependencies(
                project_path=codebase_path,
                mode=config.security_config["package_validation"]["mode"],
                whitelist_path=config.security_config["package_validation"]["whitelist_path"],
                save_report=False
            )
            
            return jsonify(results["packages"])
            
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
        
        # NEW: Update security settings
        if 'security' in data['settings']:
            if not project_config:
                project_config = {}
            
            if 'security' not in project_config:
                project_config['security'] = {}
                
            project_config['security'].update(data['settings']['security'])
            
        # Save updated config
        config.save_project_config(project_id, project_config)
    
    return jsonify({"success": True})

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

if __name__ == '__main__':
    app.run(debug=True)
