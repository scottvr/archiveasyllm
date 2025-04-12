#!/usr/bin/env python3
"""
Diagnostic tool for ArchiveAsyLLM.

This script checks the system setup and identifies potential issues with
dependencies, database connections, and configuration.
"""
import os
import sys
import time
import logging
import importlib
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("diagnostic")

def check_python_version():
    """Check Python version."""
    import platform
    version = platform.python_version()
    logger.info(f"Python version: {version}")
    
    if sys.version_info < (3, 8):
        logger.warning("Python 3.8 or higher is recommended")
        return False
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "flask",
        "neo4j",
        "faiss-cpu",
        "sentence-transformers",
        "numpy",
        "requests",
        "anthropic",
        "openai",
        "python-dotenv"
    ]
    
    missing = []
    problematic = []
    
    logger.info("Checking dependencies...")
    for package in required_packages:
        try:
            module = importlib.import_module(package.replace("-", "_"))
            logger.info(f"✓ {package} - installed version: {getattr(module, '__version__', 'unknown')}")
        except ImportError:
            logger.error(f"✗ {package} - not installed")
            missing.append(package)
        except Exception as e:
            logger.error(f"✗ {package} - error checking: {e}")
            problematic.append(package)
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.info("Install missing dependencies with: pip install " + " ".join(missing))
    
    if problematic:
        logger.warning(f"Problematic dependencies: {', '.join(problematic)}")
    
    return len(missing) == 0

def check_environment_vars():
    """Check environment variables."""
    logger.info("Checking environment variables...")
    
    # Try to load from .env
    try:
        from dotenv import load_dotenv
        env_file = Path(__file__).parent / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from {env_file}")
        else:
            logger.warning(f".env file not found at {env_file}")
    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env file loading")
    
    required_vars = [
        "GRAPH_DB_URL",
        "GRAPH_DB_USER",
        "GRAPH_DB_PASSWORD",
    ]
    
    # Either ANTHROPIC_API_KEY or OPENAI_API_KEY is required
    llm_vars = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY"
    ]
    
    missing = []
    for var in required_vars:
        if not os.environ.get(var):
            logger.error(f"✗ {var} not set")
            missing.append(var)
        else:
            logger.info(f"✓ {var} is set" + (f" to {os.environ.get(var)}" if var not in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GRAPH_DB_PASSWORD"] else ""))
    
    # Check if at least one LLM provider API key is set
    llm_keys_set = False
    for var in llm_vars:
        if os.environ.get(var):
            llm_keys_set = True
            logger.info(f"✓ {var} is set")
        else:
            logger.info(f"✗ {var} not set")
    
    if not llm_keys_set:
        logger.error("Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set")
        missing.append("LLM API key")
    
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        logger.info("Create a .env file or set these variables in your environment")
        return False
    
    return True

def check_neo4j():
    """Check if Neo4j is accessible."""
    logger.info("Checking Neo4j connection...")
    
    try:
        from neo4j import GraphDatabase
        
        neo4j_url = os.environ.get("GRAPH_DB_URL", "bolt://localhost:7687")
        neo4j_user = os.environ.get("GRAPH_DB_USER", "neo4j")
        neo4j_password = os.environ.get("GRAPH_DB_PASSWORD", "")
        
        # Fail fast if no password
        if not neo4j_password:
            logger.error("Neo4j password not set")
            return False
        
        # Try to connect
        start_time = time.time()
        driver = GraphDatabase.driver(
            neo4j_url, 
            auth=(neo4j_user, neo4j_password),
            connection_timeout=5,  # Short timeout for diagnostics
            connection_acquisition_timeout=5
        )
        
        logger.info(f"Created Neo4j driver in {time.time() - start_time:.2f}s")
        
        # Run a simple query
        with driver.session() as session:
            query_start = time.time()
            result = session.run("RETURN 1 as value")
            value = result.single()["value"]
            
            if value != 1:
                raise Exception(f"Unexpected result: {value}")
            
            logger.info(f"Query executed successfully in {time.time() - query_start:.2f}s")
        
        # Check Neo4j version
        with driver.session() as session:
            result = session.run("CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition")
            record = result.single()
            logger.info(f"Neo4j version: {record['name']} {record['versions'][0]} {record['edition']}")
        
        driver.close()
        logger.info("Neo4j connection successful")
        return True
        
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        
        # Provide helpful tips
        import platform
        if platform.system() == "Windows":
            logger.info("On Windows:")
            logger.info("1. Make sure Neo4j is installed and running")
            logger.info("2. Check if Neo4j service is running: neo4j.bat status")
            logger.info("3. Try starting Neo4j: neo4j.bat start")
            logger.info("4. Verify your credentials in the .env file")
        else:
            logger.info("On Linux/Mac:")
            logger.info("1. Make sure Neo4j is installed and running")
            logger.info("2. Check if Neo4j service is running: systemctl status neo4j")
            logger.info("3. Try starting Neo4j: sudo systemctl start neo4j")
            logger.info("4. Verify your credentials in the .env file")
        
        logger.info("If you're using Neo4j Desktop, make sure the database is started")
        logger.info("Check Neo4j Browser at http://localhost:7474/ to verify your credentials")
        
        return False

def check_embedding_model():
    """Check if the embedding model can be loaded."""
    logger.info("Checking embedding model...")
    
    try:
        # Try importing required libraries
        import torch
        from sentence_transformers import SentenceTransformer
        
        # Check CUDA availability
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Try loading a small model
        start_time = time.time()
        logger.info("Loading embedding model (this may take a moment)...")
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # Use CPU for diagnostics
        logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
        
        # Try encoding a simple text
        text = "This is a test sentence for embedding model diagnostics."
        encode_start = time.time()
        embedding = model.encode(text)
        logger.info(f"Text encoded in {time.time() - encode_start:.2f}s")
        logger.info(f"Embedding shape: {embedding.shape}")
        
        logger.info("Embedding model check successful")
        return True
        
    except Exception as e:
        logger.error(f"Error checking embedding model: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Provide helpful tips
        logger.info("Possible solutions:")
        logger.info("1. Make sure sentence-transformers is properly installed: pip install -U sentence-transformers")
        logger.info("2. If you're using CUDA, make sure PyTorch is installed with CUDA support")
        logger.info("3. Try using a different model or running on CPU only")
        
        return False

def check_llm_api():
    """Check if LLM API keys are valid."""
    # First determine which API key to check
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if anthropic_key:
        logger.info("Checking Anthropic API connection...")
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            
            # Try a simple completion
            start_time = time.time()
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Say hello"}]
            )
            logger.info(f"Anthropic API request completed in {time.time() - start_time:.2f}s")
            logger.info(f"Response received: {response.content[0].text[:30]}...")
            logger.info("Anthropic API connection successful")
            return True
        except Exception as e:
            logger.error(f"Anthropic API check failed: {e}")
            logger.info("Check your ANTHROPIC_API_KEY and internet connection")
            return False
    
    elif openai_key:
        logger.info("Checking OpenAI API connection...")
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            
            # Try a simple completion
            start_time = time.time()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                max_tokens=10,
                messages=[{"role": "user", "content": "Say hello"}]
            )
            logger.info(f"OpenAI API request completed in {time.time() - start_time:.2f}s")
            logger.info(f"Response received: {response.choices[0].message.content[:30]}...")
            logger.info("OpenAI API connection successful")
            return True
        except Exception as e:
            logger.error(f"OpenAI API check failed: {e}")
            logger.info("Check your OPENAI_API_KEY and internet connection")
            return False
    
    else:
        logger.error("No LLM API keys found")
        return False

def check_disk_space():
    """Check available disk space."""
    logger.info("Checking disk space...")
    
    try:
        import shutil
        
        # Check space in the current directory
        total, used, free = shutil.disk_usage(".")
        
        # Convert to GB for readability
        total_gb = total / (1024 ** 3)
        used_gb = used / (1024 ** 3)
        free_gb = free / (1024 ** 3)
        
        logger.info(f"Total disk space: {total_gb:.2f} GB")
        logger.info(f"Used disk space: {used_gb:.2f} GB")
        logger.info(f"Free disk space: {free_gb:.2f} GB")
        
        # Check if we have enough space (at least p500 MB)
        if free_gb < 0.5:
            logger.warning("Low disk space: less than 500 MB available")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error checking disk space: {e}")
        return False

def check_project_structure():
    """Check if the project structure is correct."""
    logger.info("Checking project structure...")
    
    # Required directories
    required_dirs = [
        "data",
        "data/projects",
        "data/chats",
        "data/vector_indexes",
        "archiveasy",
        "archiveasy/templates",
        "archiveasy/static"
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        dir_path = Path(__file__).parent / directory
        if not dir_path.exists():
            logger.error(f"✗ Directory not found: {directory}")
            missing_dirs.append(directory)
        else:
            logger.info(f"✓ Directory exists: {directory}")
    
    if missing_dirs:
        logger.error(f"Missing directories: {', '.join(missing_dirs)}")
        logger.info("Create the missing directories or reinstall the application")
        return False
    
    return True

def run_diagnostics():
    """Run all diagnostic checks."""
    logger.info("=== ArchiveAsyLLM Diagnostics ===")
    
    results = {
        "Python version": check_python_version(),
        "Dependencies": check_dependencies(),
        "Environment variables": check_environment_vars(),
        "Project structure": check_project_structure(),
        "Disk space": check_disk_space(),
        "Neo4j connection": check_neo4j(),
        "Embedding model": check_embedding_model(),
        "LLM API": check_llm_api()
    }
    
    logger.info("\n=== Diagnostic Summary ===")
    
    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status} - {check}")
        all_passed = all_passed and passed
    
    if all_passed:
        logger.info("\nAll checks passed! The system should be working correctly.")
    else:
        logger.warning("\nSome checks failed. Review the issues above and take corrective action.")
    
    return all_passed

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ArchiveAsyLLM Diagnostic Tool")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    success = run_diagnostics()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
