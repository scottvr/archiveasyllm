#!/usr/bin/env python3
"""
ArchiveAsyLLM API Client
A Python client for interacting with the ArchiveAsyLLM API.
"""
# crumb: clients\python\api-client.py
import requests
import json
from typing import Dict, Any, List, Optional, Union
import os

class ArchivistClient:
    """
    Client for interacting with the ArchiveAsyLLM API.
    """
    
    def __init__(self, api_url: str, api_key: str):
        """
        Initialize the ArchiveAsyLLM API client.
        
        Args:
            api_url: Base URL for the ArchiveAsyLLM API (e.g., "http://localhost:5000/api/v1")
            api_key: API key for authentication
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    def _request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None, 
                params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an API request.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request data (for POST/PUT)
            params: Query parameters (for GET)
            
        Returns:
            JSON response
            
        Raises:
            ArchivistAPIError: If the API returns an error
        """
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        
        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, json=data)
            elif method == "PUT":
                response = requests.put(url, headers=self.headers, json=data)
            elif method == "DELETE":
                response = requests.delete(url, headers=self.headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check for errors
            if not response.ok:
                error_info = response.json() if response.content else {"error": "Unknown error"}
                raise ArchivistAPIError(
                    f"API error ({response.status_code}): {error_info.get('error', 'Unknown error')}",
                    status_code=response.status_code,
                    error_info=error_info
                )
            
            # Parse and return JSON response
            if response.content:
                return response.json()
            else:
                return {"success": True}
                
        except requests.RequestException as e:
            raise ArchivistAPIError(f"Request failed: {str(e)}")
    
    # Project methods
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """
        List all projects.
        
        Returns:
            List of project dictionaries
        """
        return self._request("GET", "projects")
    
    def create_project(self, name: str, description: str = "", 
                      codebase_path: Optional[str] = None,
                      excluded_dirs: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new project.
        
        Args:
            name: Project name
            description: Project description
            codebase_path: Path to codebase (optional)
            excluded_dirs: Comma-separated list of directories to exclude (optional)
            
        Returns:
            Created project dictionary
        """
        data = {
            "name": name,
            "description": description
        }
        
        if codebase_path:
            data["codebase_path"] = codebase_path
        
        if excluded_dirs:
            data["excluded_dirs"] = excluded_dirs
        
        return self._request("POST", "projects", data=data)
    
    def get_project(self, project_id: str) -> Dict[str, Any]:
        """
        Get project details.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Project dictionary
        """
        return self._request("GET", f"projects/{project_id}")
    
    def update_project(self, project_id: str, name: Optional[str] = None, 
                      description: Optional[str] = None,
                      settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update project details.
        
        Args:
            project_id: Project identifier
            name: New project name (optional)
            description: New project description (optional)
            settings: Project settings (optional)
            
        Returns:
            Updated project dictionary
        """
        data = {}
        
        if name is not None:
            data["name"] = name
        
        if description is not None:
            data["description"] = description
        
        if settings is not None:
            data["settings"] = settings
        
        return self._request("PUT", f"projects/{project_id}", data=data)
    
    def delete_project(self, project_id: str) -> Dict[str, Any]:
        """
        Delete a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Success confirmation
        """
        return self._request("DELETE", f"projects/{project_id}")
    
    # Chat methods
    
    def list_chats(self, project_id: str) -> List[Dict[str, Any]]:
        """
        List all chats for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of chat dictionaries
        """
        return self._request("GET", f"projects/{project_id}/chats")
    
    def create_chat(self, project_id: str, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new chat in a project.
        
        Args:
            project_id: Project identifier
            name: Chat name (optional)
            
        Returns:
            Created chat dictionary
        """
        data = {}
        
        if name is not None:
            data["name"] = name
        
        return self._request("POST", f"projects/{project_id}/chats", data=data)
    
    def get_chat(self, chat_id: str) -> Dict[str, Any]:
        """
        Get chat details.
        
        Args:
            chat_id: Chat identifier
            
        Returns:
            Chat dictionary
        """
        return self._request("GET", f"chats/{chat_id}")
    
    def list_messages(self, chat_id: str) -> List[Dict[str, Any]]:
        """
        List all messages in a chat.
        
        Args:
            chat_id: Chat identifier
            
        Returns:
            List of message dictionaries
        """
        return self._request("GET", f"chats/{chat_id}/messages")
    
    def send_message(self, chat_id: str, content: str) -> Dict[str, Any]:
        """
        Send a message to the LLM and get a response.
        
        Args:
            chat_id: Chat identifier
            content: Message content
            
        Returns:
            Response dictionary with message and consistency issues
        """
        data = {
            "content": content
        }
        
        return self._request("POST", f"chats/{chat_id}/messages", data=data)
    
    # Codebase methods
    
    def analyze_codebase(self, project_id: str, codebase_path: Optional[str] = None,
                        excluded_dirs: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a codebase for a project.
        
        Args:
            project_id: Project identifier
            codebase_path: Path to codebase (optional if already set)
            excluded_dirs: Comma-separated list of directories to exclude (optional)
            
        Returns:
            Analysis results
        """
        data = {}
        
        if codebase_path is not None:
            data["codebase_path"] = codebase_path
        
        if excluded_dirs is not None:
            data["excluded_dirs"] = excluded_dirs
        
        return self._request("POST", f"projects/{project_id}/codebase", data=data)
    
    # Knowledge methods
    
    def get_knowledge(self, project_id: str, knowledge_type: str, 
                     search_query: Optional[str] = None,
                     limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get knowledge elements from the knowledge graph.
        
        Args:
            project_id: Project identifier
            knowledge_type: Type of knowledge (decisions, patterns, entities, relationships, search)
            search_query: Search query (required for 'search' type, optional for others)
            limit: Maximum number of results (default: 1000)
            offset: Number of results to skip (default: 0)
            
        Returns:
            List of knowledge elements
        """
        params = {
            "limit": limit,
            "offset": offset
        }
        
        if search_query is not None:
            params["q"] = search_query
        
        return self._request("GET", f"projects/{project_id}/knowledge/{knowledge_type}", params=params)
    
    def get_decisions(self, project_id: str, search_query: Optional[str] = None,
                     limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get architectural decisions.
        
        Args:
            project_id: Project identifier
            search_query: Search query (optional)
            limit: Maximum number of results (default: 1000)
            offset: Number of results to skip (default: 0)
            
        Returns:
            List of decisions
        """
        return self.get_knowledge(project_id, "decisions", search_query, limit, offset)
    
    def get_patterns(self, project_id: str, search_query: Optional[str] = None,
                    limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get design patterns.
        
        Args:
            project_id: Project identifier
            search_query: Search query (optional)
            limit: Maximum number of results (default: 1000)
            offset: Number of results to skip (default: 0)
            
        Returns:
            List of patterns
        """
        return self.get_knowledge(project_id, "patterns", search_query, limit, offset)
    
    def get_entities(self, project_id: str, search_query: Optional[str] = None,
                    limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get code entities.
        
        Args:
            project_id: Project identifier
            search_query: Search query (optional)
            limit: Maximum number of results (default: 1000)
            offset: Number of results to skip (default: 0)
            
        Returns:
            List of entities
        """
        return self.get_knowledge(project_id, "entities", search_query, limit, offset)
    
    def get_relationships(self, project_id: str, search_query: Optional[str] = None,
                         limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get entity relationships.
        
        Args:
            project_id: Project identifier
            search_query: Search query (optional)
            limit: Maximum number of results (default: 1000)
            offset: Number of results to skip (default: 0)
            
        Returns:
            List of relationships
        """
        return self.get_knowledge(project_id, "relationships", search_query, limit, offset)
    
    def search_knowledge(self, project_id: str, search_query: str,
                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search across all knowledge.
        
        Args:
            project_id: Project identifier
            search_query: Search query
            limit: Maximum number of results (default: 10)
            
        Returns:
            List of search results
        """
        return self.get_knowledge(project_id, "search", search_query, limit)
    
    # Consistency checking
    
    def check_consistency(self, project_id: str, content: str, 
                         artifacts: Optional[List[Dict[str, Any]]] = None,
                         store_knowledge: bool = False) -> Dict[str, Any]:
        """
        Check text for consistency with existing knowledge.
        
        Args:
            project_id: Project identifier
            content: Text content to check
            artifacts: List of artifacts (optional)
            store_knowledge: Whether to store extracted knowledge (default: False)
            
        Returns:
            Consistency check results
        """
        data = {
            "content": content,
            "store_knowledge": store_knowledge
        }
        
        if artifacts is not None:
            data["artifacts"] = artifacts
        
        return self._request("POST", f"projects/{project_id}/consistency", data=data)


class ArchivistAPIError(Exception):
    """Exception raised for ArchiveAsyLLM API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                error_info: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            status_code: HTTP status code (optional)
            error_info: Additional error information (optional)
        """
        self.message = message
        self.status_code = status_code
        self.error_info = error_info
        super().__init__(self.message)


# Usage example
if __name__ == "__main__":
    import sys
    
    # Get API key from environment or command line
    api_key = os.environ.get("ARCHIVIST_API_KEY") or "dev-api-key"
    api_url = os.environ.get("ARCHIVIST_API_URL") or "http://localhost:5000/api/v1"
    
    client = ArchivistClient(api_url, api_key)
    
    if len(sys.argv) < 2:
        print("Usage: python archiveasy_client.py <command>")
        print("Commands: list-projects, create-project, analyze-codebase")
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "list-projects":
            projects = client.list_projects()
            print(f"Found {len(projects)} projects:")
            for project in projects:
                print(f"- {project['name']} (ID: {project['id']})")
                print(f"  Description: {project['description']}")
                print()
                
        elif command == "create-project":
            if len(sys.argv) < 3:
                print("Usage: python archiveasy_client.py create-project <name> [description] [codebase_path]")
                sys.exit(1)
                
            name = sys.argv[2]
            description = sys.argv[3] if len(sys.argv) > 3 else ""
            codebase_path = sys.argv[4] if len(sys.argv) > 4 else None
            
            project = client.create_project(name, description, codebase_path)
            print(f"Created project: {project['name']} (ID: {project['id']})")
            
        elif command == "analyze-codebase":
            if len(sys.argv) < 3:
                print("Usage: python archiveasy_client.py analyze-codebase <project_id> [codebase_path]")
                sys.exit(1)
                
            project_id = sys.argv[2]
            codebase_path = sys.argv[3] if len(sys.argv) > 3 else None
            
            result = client.analyze_codebase(project_id, codebase_path)
            print(f"Analysis complete: {result['message']}")
            print(f"Files processed: {result['stats']['files_processed']}")
            print(f"Entities extracted: {result['stats']['entities_extracted']}")
            print(f"Decisions identified: {result['stats']['decisions_extracted']}")
            
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
            
    except ArchivistAPIError as e:
        print(f"Error: {e.message}")
        sys.exit(1)
