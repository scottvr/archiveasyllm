#!/usr/bin/env python3
"""
ArchiveasyLLM - A framework for maintaining LLM reasoning and consistency.
"""
import os
from flask import Flask, render_template, request, jsonify, session
from datetime import datetime

# Add src to Python path if needed
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from archiveasy.llm.client import LLMClient
from archiveasy.memory.graph import KnowledgeGraph
from archiveasy.memory.vector import VectorStore
from archiveasy.analyzer.consistency import ConsistencyChecker
from archiveasy.models.chat import Chat, Message
from archiveasy.models.project import Project
from archiveasy.api.routes import api_bp, init_api
import config

app = Flask(__name__, 
           template_folder='src/archiveasy/templates',
           static_folder='src/archiveasy/static')
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# Initialize components
llm_client = LLMClient.from_config(config.llm_config)
knowledge_graph = KnowledgeGraph(config.graph_db_url)
vector_store = VectorStore(config.vector_db_config)
consistency_checker = ConsistencyChecker(knowledge_graph, vector_store)

# Make components available to API
app.config['llm_client'] = llm_client
app.config['knowledge_graph'] = knowledge_graph
app.config['vector_store'] = vector_store
app.config['consistency_checker'] = consistency_checker

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

# Rest of your app.py implementation
# ...

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
