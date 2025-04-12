#!/usr/bin/env python3
"""
Command-line interface for ArchiveAsyLLM.
"""
import sys
import os
import argparse
import logging
import time
from datetime import datetime
import json
import uuid
from typing import Dict, List, Optional

from archiveasy.memory.graph import KnowledgeGraph
from archiveasy.memory.vector import VectorStore
from archiveasy.analyzer.codebase import CodebaseAnalyzer
from archiveasy.models.project import Project
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_project(args):
    """
    Initialize a new project with an existing codebase.
    
    Args:
        args: Command-line arguments
    """
    # Configure logging based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Make sure all loggers output to console
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)
        
        # Configure specific loggers that might be causing issues
        logging.getLogger('neo4j').setLevel(logging.DEBUG)
        logging.getLogger('sentence_transformers').setLevel(logging.DEBUG)
        logging.getLogger('archiveasy.memory.vector').setLevel(logging.DEBUG)
        logging.getLogger('archiveasy.memory.graph').setLevel(logging.DEBUG)
        
        logger.debug("Debug logging enabled")
    
    start_time = time.time()
    project_name = args.name
    codebase_path = args.path
    description = args.description or f"Project initialized from {codebase_path}"
    
    logger.info(f"Initializing project '{project_name}' with codebase at {codebase_path}")
    
    # Validate codebase path
    if not os.path.isdir(codebase_path):
        logger.error(f"Codebase path does not exist or is not a directory: {codebase_path}")
        sys.exit(1)
    
    # Create project
    project_id = str(uuid.uuid4())
    logger.info(f"Generated project ID: {project_id}")
    
    # Create project directory
    project_dir = os.path.join("./data/projects", project_id)
    os.makedirs(project_dir, exist_ok=True)
    logger.info(f"Created project directory: {project_dir}")
    
    # Save project metadata
    project_metadata = {
        "id": project_id,
        "name": project_name,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "codebase_path": os.path.abspath(codebase_path)
    }
    
    with open(os.path.join(project_dir, "metadata.json"), 'w') as f:
        json.dump(project_metadata, f, indent=2)
    
    logger.info(f"Created project: {project_name} (ID: {project_id})")
    
    # Check Neo4j connection first
    logger.info("Checking Neo4j connection...")
    if not check_neo4j_connection(config.graph_db_url, config.graph_db_user, config.graph_db_password):
        logger.error("Neo4j connection failed. Please make sure Neo4j is running and accessible.")
        logger.error(f"Configuration: URL={config.graph_db_url}, User={config.graph_db_user}")
        logger.error("Check your .env file or environment variables for Neo4j settings.")
        sys.exit(1)
        
    # Initialize knowledge systems
    logger.info("Initializing Knowledge Graph connection...")
    try:
        knowledge_graph = KnowledgeGraph(config.graph_db_url, config.graph_db_user, config.graph_db_password)
        logger.info("Knowledge Graph connection established successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        logger.error("Make sure Neo4j is running and accessible")
        sys.exit(1)
    
    logger.info("Initializing Vector Store...")
    try:
        vector_store = VectorStore(config.vector_db_config)
        logger.info("Vector Store initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Vector Store: {e}")
        sys.exit(1)
    
    # Initialize analyzer
    logger.info("Creating CodebaseAnalyzer...")
    analyzer = CodebaseAnalyzer(knowledge_graph, vector_store)
    
    # Analyze codebase
    logger.info(f"Starting codebase analysis: {codebase_path}")
    try:
        stats = analyzer.analyze_codebase(
            codebase_path, 
            project_id, 
            excluded_dirs=args.exclude.split(",") if args.exclude else None,
            max_files=args.max_files
        )

        analysis_duration = time.time() - start_time
        logger.info(f"Analysis completed in {analysis_duration:.2f} seconds")
        
        # Save analysis stats
        with open(os.path.join(project_dir, "analysis_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Analysis complete. Processed {stats['files_processed']} files.")
        logger.info(f"Extracted {stats['entities_extracted']} entities, {stats['relationships_extracted']} relationships")
        logger.info(f"Identified {stats['decisions_extracted']} architectural decisions and {stats['patterns_extracted']} design patterns")
        
        # Print language breakdown
        if stats["languages"]:
            logger.info("Language breakdown:")
            for lang, count in stats["languages"].items():
                logger.info(f"  - {lang}: {count} files")
        
        # Save project config
        project_config = {
            "llm": {
                "provider": config.llm_config["provider"],
                "model": config.llm_config["model"],
                "temperature": 0.7
            },
            "memory": {
                "extract_patterns": True,
                "extract_decisions": True,
                "extract_entities": True,
                "consistency_checking": True
            },
            "codebase": {
                "path": os.path.abspath(codebase_path),
                "last_analyzed": datetime.now().isoformat(),
                "stats": stats
            }
        }
        
        config.save_project_config(project_id, project_config)
        
        logger.info(f"Project initialized successfully. Project ID: {project_id}")
        logger.info(f"To start using this project, run: flask run")
        logger.info(f"Then open http://localhost:5000/project/{project_id} in your browser")
        
    except Exception as e:
        logger.error(f"Error analyzing codebase: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

def list_projects(args):
    """
    List all projects.
    
    Args:
        args: Command-line arguments
    """
    projects_dir = "./data/projects"
    
    if not os.path.isdir(projects_dir):
        logger.info("No projects found.")
        return
    
    projects = []
    
    for project_id in os.listdir(projects_dir):
        metadata_file = os.path.join(projects_dir, project_id, "metadata.json")
        
        if os.path.isfile(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                projects.append(metadata)
            except Exception as e:
                logger.warning(f"Error loading project {project_id}: {e}")
    
    if not projects:
        logger.info("No projects found.")
        return
    
    logger.info(f"Found {len(projects)} projects:")
    
    for project in projects:
        logger.info(f"  - {project['name']} (ID: {project['id']})")
        logger.info(f"    Description: {project['description']}")
        logger.info(f"    Created: {project['created_at']}")
        if "codebase_path" in project:
            logger.info(f"    Codebase: {project['codebase_path']}")
        logger.info("")

def update_codebase(args):
    """
    Update project with changes from codebase.
    
    Args:
        args: Command-line arguments
    """
    project_id = args.project_id
    
    # Validate project
    projects_dir = "./data/projects"
    project_dir = os.path.join(projects_dir, project_id)
    
    if not os.path.isdir(project_dir):
        logger.error(f"Project not found: {project_id}")
        sys.exit(1)
    
    # Load project metadata
    metadata_file = os.path.join(project_dir, "metadata.json")
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Error loading project metadata: {e}")
        sys.exit(1)
    
    # Get codebase path
    codebase_path = args.path or metadata.get("codebase_path")
    
    if not codebase_path:
        logger.error("Codebase path not found in project metadata. Please specify with --path.")
        sys.exit(1)
    
    if not os.path.isdir(codebase_path):
        logger.error(f"Codebase path does not exist or is not a directory: {codebase_path}")
        sys.exit(1)
    
    logger.info(f"Updating project {metadata['name']} (ID: {project_id}) from codebase: {codebase_path}")
    
    # Initialize knowledge systems
    knowledge_graph = KnowledgeGraph(config.graph_db_url, config.graph_db_user, config.graph_db_password)
    vector_store = VectorStore(config.vector_db_config)
    
    # Initialize analyzer
    analyzer = CodebaseAnalyzer(knowledge_graph, vector_store)
    
    # Analyze codebase
    try:
        stats = analyzer.analyze_codebase(codebase_path, project_id, excluded_dirs=args.exclude.split(",") if args.exclude else None)
        
        # Save analysis stats
        with open(os.path.join(project_dir, "analysis_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Update complete. Processed {stats['files_processed']} files.")
        logger.info(f"Extracted {stats['entities_extracted']} entities, {stats['relationships_extracted']} relationships")
        logger.info(f"Identified {stats['decisions_extracted']} architectural decisions and {stats['patterns_extracted']} design patterns")
        
        # Update project config
        project_config = config.load_project_config(project_id)
        
        if "codebase" not in project_config:
            project_config["codebase"] = {}
        
        project_config["codebase"].update({
            "path": os.path.abspath(codebase_path),
            "last_analyzed": datetime.now().isoformat(),
            "stats": stats
        })
        
        config.save_project_config(project_id, project_config)
        
        logger.info(f"Project updated successfully.")
        
    except Exception as e:
        logger.error(f"Error analyzing codebase: {e}")
        sys.exit(1)

def check_neo4j_connection(url: str, user: str, password: str) -> bool:
    """
    Check if Neo4j database is accessible.
    
    Args:
        url: Neo4j database URL
        user: Neo4j username
        password: Neo4j password
        
    Returns:
        True if the connection is successful
    """
    import time
    from neo4j import GraphDatabase
    
    logger.info(f"Testing Neo4j connection to {url}")
    start_time = time.time()
    
    try:
        # Create driver with short timeout
        driver = GraphDatabase.driver(
            url, 
            auth=(user, password),
            connection_timeout=5,
            connection_acquisition_timeout=10
        )
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 1 as value")
            value = result.single()["value"]
            if value != 1:
                raise Exception("Unexpected test result")
        
        # Close driver
        driver.close()
        
        connect_time = time.time() - start_time
        logger.info(f"Neo4j connection successful in {connect_time:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        logger.error("Please make sure Neo4j is running and accessible")
        
        # Provide platform-specific guidance
        import platform
        if platform.system() == "Windows":
            logger.info("On Windows, if using Neo4j Desktop, make sure the database is started.")
            logger.info("Neo4j service commands: ")
            logger.info("  - Start: neo4j.bat start")
            logger.info("  - Status: neo4j.bat status")
        else:  # Linux/Mac
            logger.info("On Linux/Mac: ")
            logger.info("  - Start service: sudo service neo4j start")
            logger.info("  - Check status: sudo service neo4j status")
        
        return False

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="ArchiveAsyLLM Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Initialize project command
    init_parser = subparsers.add_parser("init", help="Initialize a new project with an existing codebase")
    init_parser.add_argument("name", help="Project name")
    init_parser.add_argument("path", help="Path to codebase")
    init_parser.add_argument("--description", help="Project description")
    init_parser.add_argument("--exclude", help="Comma-separated list of directories to exclude")
    init_parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    init_parser.add_argument("--max-files", type=int, help="Maximum number of files to process (for debugging)")

    # List projects command
    list_parser = subparsers.add_parser("list", help="List all projects")
    
    # Update codebase command
    update_parser = subparsers.add_parser("update", help="Update project with changes from codebase")
    update_parser.add_argument("project_id", help="Project ID")
    update_parser.add_argument("--path", help="Path to codebase (optional if already set)")
    update_parser.add_argument("--exclude", help="Comma-separated list of directories to exclude")
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_project(args)
    elif args.command == "list":
        list_projects(args)
    elif args.command == "update":
        update_codebase(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
