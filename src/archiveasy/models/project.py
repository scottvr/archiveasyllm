"""
Project data model for ArchiveasyLLM.
"""
# crumb: models\project.py
import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

class Project:
    """
    Project data model representing a collection of chats and knowledge.
    """
    
    data_dir = Path("./data/projects")
    
    def __init__(self, id: str, name: str, description: str = "", 
                 created_at: Optional[datetime] = None,
                 codebase_path: Optional[str] = None,
                 codebase_stats: Optional[Dict[str, Any]] = None):
        """
        Initialize a project.
        
        Args:
            id: Unique project identifier
            name: Project name
            description: Project description
            created_at: Creation timestamp
            codebase_path: Path to associated codebase
            codebase_stats: Statistics from codebase analysis
        """
        self.id = id
        self.name = name
        self.description = description
        self.created_at = created_at or datetime.now()
        self.codebase_path = codebase_path
        self.codebase_stats = codebase_stats
    
    @classmethod
    def create(cls, name: str, description: str = "") -> 'Project':
        """
        Create a new project.
        
        Args:
            name: Project name
            description: Project description
            
        Returns:
            New Project instance
        """
        # Generate unique ID
        project_id = str(uuid.uuid4())
        
        # Create project directory
        project_dir = cls.data_dir / project_id
        os.makedirs(project_dir, exist_ok=True)
        
        # Create project instance
        project = cls(
            id=project_id,
            name=name,
            description=description,
            created_at=datetime.now()
        )
        
        # Save to disk
        project._save()
        
        return project
    
    @classmethod
    def get(cls, project_id: str) -> Optional['Project']:
        """
        Get a project by ID.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Project instance or None if not found
        """
        metadata_file = cls.data_dir / project_id / "metadata.json"
        
        if not os.path.exists(metadata_file):
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            # Parse created_at timestamp
            created_at = datetime.fromisoformat(data.get('created_at')) if 'created_at' in data else None
            
            return cls(
                id=data.get('id'),
                name=data.get('name'),
                description=data.get('description', ''),
                created_at=created_at,
                codebase_path=data.get('codebase_path'),
                codebase_stats=data.get('codebase_stats')
            )
        except Exception as e:
            print(f"Error loading project {project_id}: {e}")
            return None
    
    @classmethod
    def list_all(cls) -> List['Project']:
        """
        List all projects.
        
        Returns:
            List of Project instances
        """
        projects = []
        
        if not os.path.exists(cls.data_dir):
            os.makedirs(cls.data_dir, exist_ok=True)
            return projects
        
        for project_id in os.listdir(cls.data_dir):
            project_dir = cls.data_dir / project_id
            
            if not os.path.isdir(project_dir):
                continue
            
            project = cls.get(project_id)
            if project:
                projects.append(project)
        
        # Sort by created_at (newest first)
        projects.sort(key=lambda p: p.created_at, reverse=True)
        
        return projects
    
    def update(self, **kwargs) -> 'Project':
        """
        Update project attributes.
        
        Args:
            **kwargs: Attributes to update
            
        Returns:
            Updated Project instance
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Save changes
        self._save()
        
        return self
    
    def delete(self) -> bool:
        """
        Delete the project.
        
        Returns:
            True if successful
        """
        import shutil
        
        project_dir = self.data_dir / self.id
        
        if os.path.exists(project_dir):
            try:
                shutil.rmtree(project_dir)
                return True
            except Exception as e:
                print(f"Error deleting project {self.id}: {e}")
        
        return False
    
    def _save(self) -> None:
        """Save project metadata to disk."""
        project_dir = self.data_dir / self.id
        os.makedirs(project_dir, exist_ok=True)
        
        metadata_file = project_dir / "metadata.json"
        
        data = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
        }
        
        if self.codebase_path:
            data['codebase_path'] = self.codebase_path
        
        if self.codebase_stats:
            data['codebase_stats'] = self.codebase_stats
        
        with open(metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary of project attributes
        """
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'codebase_path': self.codebase_path,
            'codebase_stats': self.codebase_stats
        }
