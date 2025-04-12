"""
Configuration module for ArchiveasyLLM.

This module handles loading configuration from environment variables
and provides project/chat-specific configuration handling.

Configuration is loaded from:
1. Environment variables (highest priority)
2. .env file if available
3. Default values (lowest priority)
"""
import os
import json
import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Load .env file if available
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / '.env'
    load_dotenv(env_file)
    print(f"Loaded configuration from {env_file}")
except Exception as e:
    pass  # .env file is optional

# Create data directories if they don't exist
data_root = Path(__file__).parent / 'data'
for subdir in ['projects', 'chats', 'vector_indexes']:
    os.makedirs(data_root / subdir, exist_ok=True)

# Default LLM configuration
llm_config = {
    "provider": os.environ.get("LLM_PROVIDER", "anthropic"),
    "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
    "model": os.environ.get("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
    "max_tokens": int(os.environ.get("MAX_TOKENS", "4096")),
    "temperature": float(os.environ.get("TEMPERATURE", "0.7")),
}

# Alternative configuration for OpenAI
if llm_config["provider"] == "openai":
    llm_config["api_key"] = os.environ.get("OPENAI_API_KEY", "")
    llm_config["model"] = os.environ.get("OPENAI_MODEL", "gpt-4")

# Neo4j configuration
graph_db_url = os.environ.get("GRAPH_DB_URL", "bolt://localhost:7687")
graph_db_user = os.environ.get("GRAPH_DB_USER", "neo4j")
graph_db_password = os.environ.get("GRAPH_DB_PASSWORD", "")

# Vector database configuration
vector_db_config = {
    "embedding_model": os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    "dimension": 384,  # Default for all-MiniLM-L6-v2
    "index_path": os.environ.get("VECTOR_INDEX_PATH", str(data_root / "vector_indexes")),
    "distance_threshold": float(os.environ.get("DISTANCE_THRESHOLD", "0.75")),
}

# Application configuration
app_config = {
    "debug": os.environ.get("DEBUG", "False").lower() == "true",
    "host": os.environ.get("HOST", "0.0.0.0"),
    "port": int(os.environ.get("PORT", "5000")),
    "secret_key": os.environ.get("SECRET_KEY", "dev-secret-key"),
}

# API configuration
api_config = {
    "api_key": os.environ.get("API_KEY", "dev-api-key"),
}

# Security configuration (NEW)
security_config = {
    "package_validation": {
        "mode": os.environ.get("PACKAGE_VALIDATION_MODE", "verify"),  # 'whitelist' or 'verify'
        "whitelist_path": os.environ.get("PACKAGE_WHITELIST_PATH", 
                          str(Path(__file__).parent / "archiveasy" / "security" / "conf" / "package_whitelist.json")),
        "enabled": os.environ.get("ENABLE_PACKAGE_VALIDATION", "True").lower() == "true"
    },
    "code_scanning": {
        "enabled": os.environ.get("ENABLE_CODE_SCANNING", "True").lower() == "true",
        "scan_imports": os.environ.get("SCAN_IMPORTS", "True").lower() == "true",
        "scan_requirements": os.environ.get("SCAN_REQUIREMENTS", "True").lower() == "true"
    }
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
    config_path = data_root / "projects" / project_id / "config.json"
    
    if config_path.exists():
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
        },
        "security": {  # NEW
            "package_validation": security_config["package_validation"]["mode"],
            "scan_generated_code": True
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
    config_path = data_root / "chats" / chat_id / "config.json"
    
    if config_path.exists():
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
        },
        "security": {  # NEW
            "scan_generated_code": True
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
    config_dir = data_root / "projects" / project_id
    config_path = config_dir / "config.json"
    
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
    config_dir = data_root / "chats" / chat_id
    config_path = config_dir / "config.json"
    
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
        "app": app_config.copy(),
        "api": api_config.copy(),
        "security": security_config.copy()  # NEW
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
        
        # Merge security config (NEW)
        if "security" in project_config:
            if "security" not in merged:
                merged["security"] = {}
            for key, section in project_config["security"].items():
                if key not in merged["security"]:
                    merged["security"][key] = {}
                if isinstance(section, dict):
                    merged["security"][key].update(section)
                else:
                    merged["security"][key] = section
    
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
        
        # Merge security config (NEW)
        if "security" in chat_config:
            if "security" not in merged:
                merged["security"] = {}
            for key, section in chat_config["security"].items():
                if key not in merged["security"]:
                    merged["security"][key] = {}
                if isinstance(section, dict):
                    merged["security"][key].update(section)
                else:
                    merged["security"][key] = section
    
    return merged

# Validate configuration
def validate_config() -> List[str]:
    """
    Validate the current configuration and return a list of warnings.
    
    Returns:
        List of warning messages
    """
    warnings = []
    
    # Check for missing API keys
    if llm_config["provider"] == "anthropic" and not llm_config["api_key"]:
        warnings.append("Missing ANTHROPIC_API_KEY environment variable")
    
    if llm_config["provider"] == "openai" and not llm_config["api_key"]:
        warnings.append("Missing OPENAI_API_KEY environment variable")
    
    # Check for Neo4j credentials
    if not graph_db_password:
        warnings.append("Missing GRAPH_DB_PASSWORD environment variable")
    
    # Check for default API key
    if api_config["api_key"] == "dev-api-key":
        warnings.append("Using default API_KEY. For production, set a secure API_KEY")
    
    # Check for default secret key
    if app_config["secret_key"] == "dev-secret-key":
        warnings.append("Using default SECRET_KEY. For production, set a secure SECRET_KEY")
    
    return warnings

# Print configuration warnings on module load
config_warnings = validate_config()
if config_warnings:
    print("⚠️ Configuration warnings:")
    for warning in config_warnings:
        print(f"  - {warning}")
