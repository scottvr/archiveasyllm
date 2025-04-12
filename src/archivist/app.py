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
            from archivist.analyzer.codebase import CodebaseAnalyzer
            
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