#!/usr/bin/env python3
"""
ArchiveAsyLLM - A framework for maintaining LLM reasoning and consistency.
"""
import os
from flask import Flask, render_template, request, jsonify, session
from archivist.llm.client import LLMClient
from archivist.memory.graph import KnowledgeGraph
from archivist.memory.vector import VectorStore
from archivist.analyzer.consistency import ConsistencyChecker
from archivist.models.chat import Chat, Message
from archivist.models.project import Project
from archivist.api.routes import api_bp, init_api
import config

app = Flask(__name__)
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