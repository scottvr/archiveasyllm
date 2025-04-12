# ArchiveAsyLLM

ArchiveAsyLLM is a framework for maintaining consistency in LLM-driven development by tracking reasoning, patterns, and architectural decisions across conversations.

LLMs are powerful tools for code generation and problem-solving, but they have a significant limitation: they can "forget" previous design decisions or architectural patterns when code evolves over multiple prompts. ArchiveAsyLLM attempts to solve this problem by:

1. Extracting and storing architectural decisions, patterns, and reasoning
2. Building a knowledge graph of code entities and their relationships
3. Analyzing new generations for consistency with established patterns
4. Enhancing prompts with relevant context from previous conversations

## Features

- **Knowledge Extraction**: Automatically identifies patterns, decisions, and entities in LLM responses
- **Knowledge Graph Storage**: Stores extracted knowledge in a Neo4j graph database for efficient querying
- **Vector Search**: Uses FAISS for semantic search of previous conversations
- **Consistency Checking**: Identifies inconsistencies between new code and established patterns
- **Enhanced Prompts**: Augments user prompts with relevant context from the knowledge graph
- **Web Interface**: Simple Flask-based UI for projects and conversations

## Contributions

I am by no means an expert in NLP, and as you see I'm not claiming to be (though the existence of PASTAL, VisoCORE, and other tools might make it seem I pretend to be), rather I am someone who frequently finds that the solutions to problems I'd like to solve seem to be pretty squarely in that realm, so I am learning.

Please, if you are interested and ArchiveAsyLLM is (or could be) helpful and you have knowledge and expertise to share, *please* do so by submitting a PR or opening an Issue.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/archiveasyllm.git
cd archiveasyllm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. Install Neo4j Community Edition (or use Neo4j Desktop)

5. Start the application:
```bash
python app.py
```

## 📚 Usage

1. **Create a Project**: Start by creating a new project to organize your LLM conversations
2. **Create a Chat**: Within your project, create a new chat for your conversation
3. **Start Prompting**: Use the chat interface to interact with the LLM
4. **View Knowledge**: Explore extracted knowledge in the project dashboard
5. **Observe Consistency**: The system will flag any inconsistencies with previously established patterns

## 🧩 Architecture

ArchiveAsyLLM consists of several core components:

- **Flask Web App**: Simple UI for interacting with the system
- **LLM Client**: Provider-agnostic client supporting Anthropic and OpenAI
- **Knowledge Graph**: Neo4j database for storing architectural knowledge
- **Vector Store**: FAISS-based vector database for semantic search
- **Extractor**: Analyzes LLM responses to extract knowledge
- **Consistency Checker**: Identifies inconsistencies with established patterns

## 🛠️ Configuration

ArchiveAsyLLM supports multiple levels of configuration:

- **Global Configuration**: In `config.py` or environment variables
- **Project Configuration**: Project-specific settings
- **Chat Configuration**: Settings for individual chats

Configuration options include:
- LLM provider, model, and parameters
- Extraction settings (what patterns to extract)
- Consistency checking thresholds
- Knowledge graph and vector store settings

## 📊 Example Workflow

1. Start a new project for a web application
2. Discuss the architecture with the LLM
3. As the LLM suggests patterns and decisions, they're automatically extracted
4. When implementing specific components, the system reminds the LLM of previous decisions
5. If the LLM suggests something inconsistent with earlier design choices, the system flags it

## 🔄 Integration with External Tools

ArchiveAsyLLM can be extended to integrate with:

- **Version Control**: Track code changes alongside architectural decisions
- **Documentation**: Generate architecture docs from extracted knowledge
- **IDE Extensions**: Provide context and validation during development

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
