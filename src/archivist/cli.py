#!/usr/bin/env python3
"""
Command-line interface for ArchiveAsyLLM.
"""
import argparse
import os
import sys
import json
import uuid
import logging
from datetime import datetime

from archivist.memory.graph import KnowledgeGraph
from archivist.memory.vector import VectorStore
from archivist.analyzer.codebase import CodebaseAnalyzer
from archivist.models.project import Project
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
    project_name = args.name
    codebase_path = args.path
    description = args.description or f"Project initialized from {codebase_path}"
    
    # Validate codebase path
    if not os.path.isdir(codebase_path):
        logger.error(f"Codebase path does not exist or is not a directory: {codebase_path}")
        sys.exit(1)
    
    # Create project
    project_id = str(uuid.uuid4())
    
    # Create project directory
    project_dir = os.path.join("./data/projects", project_id)
    os.makedirs(project_dir, exist_ok=True)
    
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
    
    # Initialize knowledge systems
    knowledge_graph = KnowledgeGraph(config.graph_db_url, config.graph_db_user, config.graph_db_password)
    vector_store = VectorStore(config.vector_db_config)
    
    # Initialize analyzer
    analyzer = CodebaseAnalyzer(knowledge_graph, vector_store)
    
    # Analyze codebase
    logger.info(f"Analyzing codebase: {codebase_path}")
    try:
        stats = analyzer.analyze_codebase(codebase_path, project_id, excluded_dirs=args.exclude.split(",") if args.exclude else None)
        
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
