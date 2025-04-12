Getting ArchiveAsyLLM Running on Windows 10 with WSL2

## Prerequisites Setup

### Set up WSL2 with Ubuntu:
[link to wsl2 setup docs]

### Install Python dependencies in WSL:
```
sudo apt update
sudo apt install -y python3-pip python3-venv git
```

### install openjdk11-jre

### Install Neo4j in WSL (for the knowledge graph):
```
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install -y neo4j=1:5.13.0
sudo neo4j start
# Set initial password (remember this)

## ArchiveAsyLLM Setup

### Clone the repository:
```
git clone https://github.com/scottvr/archiveasyllm
cd archiveasyllm
```

### Create and activate a venv

### Install dependencies:
```
pip install -r requirements.txt
```


### Set up environment variables:

```
cp .env.example .env
# Add the following to your .env file:
ANTHROPIC_API_KEY=your_anthropic_api_key
GRAPH_DB_URL=bolt://localhost:7687
GRAPH_DB_USER=neo4j
GRAPH_DB_PASSWORD=your_password

### Initialize the database structure:

```
python -c "from archiveasy.memory.graph import KnowledgeGraph; KnowledgeGraph('bolt://localhost:7687', 'neo4j', 'your_password').init_schema()"
```

Or, use the CLI:
```
python cli.py init "ArchiveAsyLLMLLM" "./path/to/archivist-llm" --description "Self-reference project for ArchiveAsyLLMLLM"
```

### Running ArchiveAsyLLMLLM

**Start the application:**
```
python app.py
```


Access the web interface in your browser:
http://localhost:5000

Initialize with ArchiveAsyLLMLLM's own codebase:
bashpython cli.py init "ArchiveAsyLLMLLM" "$(pwd)" --description "Self-reference project for ArchiveAsyLLMLLM" --exclude "venv,.git,__pycache__,node_modules"


Testing the API

Test the API with curl:
bash# Get API documentation
curl http://localhost:5000/api/v1/ -H "X-API-Key: dev-api-key"

# List projects
curl http://localhost:5000/api/v1/projects -H "X-API-Key: dev-api-key"

Test with Python client:
bashcd clients/python
python archiveasy_client.py list-projects