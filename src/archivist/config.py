"""
Configuration module for ArchiveAsyLLM.
"""
import os
import json
from typing import Dict, Any, Optional

# Default LLM configuration
llm_config = {
    "provider": os.environ.get("LLM_PROVIDER", "anthropic"),
    "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
    "model": os.environ.get("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
    "max_tokens": 4096,
    "temperature": 0.7,
}

# Alternative configuration for OpenAI
if llm_config["provider"] == "openai":
    llm_config["api_key"] = os.environ.get("OPENAI_API_KEY", "")
    llm_config["model"] = os.environ.get("OPENAI_MODEL", "gpt-4")

# Neo4j configuration
graph_db_url = os.environ.get("GRAPH_DB_URL", "bolt://localhost:7687")
graph_db_user = os.environ.get("GRAPH_DB_USER", "")
graph_db_password = os.environ.get("GRAPH_DB_PASSWORD", "")

# Vector database configuration
vector_db_config = {
    "embedding_model": os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    "dimension": 384,  # Default for all-MiniLM-L6-v2
    "index_path": os.environ.get("VECTOR_INDEX_PATH", "./data/vector_indexes"),
    "distance_threshold": 0.75,
}

# Application configuration
app_config = {
    "debug": os.environ.get("DEBUG", "False").lower() == "true",
    "host": os.environ.get("HOST", "0.0.0.0"),
    "port": int(os.environ.get("PORT", "5000")),
    "secret_key": os.environ.get("SECRET_KEY", "dev-secret-key"),
}

# Load project-specific configurations
def load_project_config(project_id: str) -> Dict[str, Any]:
    """
    Load project-specific configuration.
    
    Args:
        project_id: Project identifier
        
    Returns:
        Project configuration dictionary
    """
    config_path = f"./data/projects/{project_id}/config.json"
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading project config: {e}")
    
    # Return default project configuration
    return {
        "llm": {
            "provider": llm_config["provider"],
            "model": llm_config["model"],
            "temperature": llm_config["temperature"]
        },
        "memory": {
            "extract_patterns": True,
            "extract_decisions": True,
            "extract_entities": True,
            "consistency_checking": True
        }
    }

# Load chat-specific configurations
def load_chat_config(chat_id: str) -> Dict[str, Any]:
    """
    Load chat-specific configuration.
    
    Args:
        chat_id: Chat identifier
        
    Returns:
        Chat configuration dictionary
    """
    config_path = f"./data/chats/{chat_id}/config.json"
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading chat config: {e}")
    
    # Return default chat configuration
    return {
        "llm": {
            "temperature": llm_config["temperature"],
            "max_tokens": llm_config["max_tokens"]
        },
        "memory": {
            "use_project_context": True,
            "use_chat_history": True,
            "include_artifacts_in_context": True
        }
    }

# Save project configuration
def save_project_config(project_id: str, config: Dict[str, Any]) -> bool:
    """
    Save project-specific configuration.
    
    Args:
        project_id: Project identifier
        config: Configuration dictionary
        
    Returns:
        True if saved successfully
    """
    config_dir = f"./data/projects/{project_id}"
    config_path = f"{config_dir}/config.json"
    
    try:
        os.makedirs(config_dir, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving project config: {e}")
        return False

# Save chat configuration
def save_chat_config(chat_id: str, config: Dict[str, Any]) -> bool:
    """
    Save chat-specific configuration.
    
    Args:
        chat_id: Chat identifier
        config: Configuration dictionary
        
    Returns:
        True if saved successfully
    """
    config_dir = f"./data/chats/{chat_id}"
    config_path = f"{config_dir}/config.json"
    
    try:
        os.makedirs(config_dir, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving chat config: {e}")
        return False

# Merge configurations (base + project + chat)
def get_merged_config(project_id: Optional[str] = None, chat_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get merged configuration combining base, project, and chat configs.
    
    Args:
        project_id: Optional project identifier
        chat_id: Optional chat identifier
        
    Returns:
        Merged configuration dictionary
    """
    merged = {
        "llm": llm_config.copy(),
        "graph_db": {
            "url": graph_db_url,
            "user": graph_db_user,
            "password": graph_db_password
        },
        "vector_db": vector_db_config.copy(),
        "app": app_config.copy()
    }
    
    # Add project config if provided
    if project_id:
        project_config = load_project_config(project_id)
        
        # Merge LLM config
        if "llm" in project_config:
            merged["llm"].update(project_config["llm"])
        
        # Merge memory config
        if "memory" in project_config:
            merged["memory"] = project_config["memory"]
    
    # Add chat config if provided
    if chat_id:
        chat_config = load_chat_config(chat_id)
        
        # Merge LLM config
        if "llm" in chat_config:
            merged["llm"].update(chat_config["llm"])
        
        # Merge memory config
        if "memory" in chat_config:
            if "memory" not in merged:
                merged["memory"] = {}
            merged["memory"].update(chat_config["memory"])
    
    return merged
