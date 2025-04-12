#!/usr/bin/env python3
"""
ArchiveAsyLLM - A framework for maintaining LLM reasoning and consistency when "vibecoding".
"""
import os
from flask import Flask, render_template, request, jsonify, session
from archivist.llm.client import LLMClient
from archivist.memory.graph import KnowledgeGraph
from archivist.memory.vector import VectorStore
from archivist.analyzer.consistency import ConsistencyChecker
from archivist.models.chat import Chat, Message
from archivist.models.project import Project
import config

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# Initialize components
llm_client = LLMClient.from_config(config.llm_config)
knowledge_graph = KnowledgeGraph(config.graph_db_url)
vector_store = VectorStore(config.vector_db_config)
consistency_checker = ConsistencyChecker(knowledge_graph, vector_store)

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
    project = Project.create(name=name, description=description)
    
    # Initialize knowledge graph for this project
    knowledge_graph.initialize_project(project.id)
    
    return jsonify({"id": project.id, "name": project.name})

@app.route('/project/<project_id>')
def view_project(project_id):
    """View a specific project and its chats."""
    project = Project.get(project_id)
    chats = Chat.list_by_project(project_id)
    return render_template('project.html', project=project, chats=chats)

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

if __name__ == '__main__':
    app.run(debug=True)
