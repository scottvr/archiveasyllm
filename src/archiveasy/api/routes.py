"""
RESTful API routes for ArchiveasyLLM with OpenAPI documentation.
"""
from flask import Blueprint, request, jsonify, current_app
from flask_restx import Api, Resource, fields, Namespace
import os
import uuid
from datetime import datetime
import json
from typing import Dict, Any, List, Optional

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
    title='ArchiveasyLLM API',
    description='API for interacting with ArchiveasyLLM knowledge and chat functionality',
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

# Rest of the code remains unchanged
# ...
