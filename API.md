# ArchiveAsyLLM API Documentation

The ArchiveAsyLLM API provides programmatic access to all core functionality of the ArchiveAsyLLM framework. This allows you to integrate architectural knowledge management and LLM consistency checking into your own applications, tools, and workflows.

## Authentication

All API requests require authentication using an API key. Include the key in the `X-API-Key` header:

```
X-API-Key: your_api_key_here
```

## Base URL

All API endpoints are prefixed with `/api/v1`.

## Response Format

All responses are in JSON format. Successful responses typically include the requested data or a success confirmation. Error responses include an `error` field with a description of what went wrong.

## Projects

Projects are the top-level containers for chats, knowledge, and codebase information.

### List Projects

```
GET /api/v1/projects
```

Returns a list of all projects.

### Create Project

```
POST /api/v1/projects
```

Create a new project.

**Request Body:**
```json
{
  "name": "Project Name",
  "description": "Project description",
  "codebase_path": "/path/to/codebase",
  "excluded_dirs": "node_modules,venv,.git"
}
```

The `codebase_path` and `excluded_dirs` fields are optional. If `codebase_path` is provided, the system will analyze the codebase and extract architectural knowledge.

### Get Project

```
GET /api/v1/projects/{project_id}
```

Get details for a specific project.

### Update Project

```
PUT /api/v1/projects/{project_id}
```

Update a project's details or settings.

**Request Body:**
```json
{
  "name": "Updated Project Name",
  "description": "Updated description",
  "settings": {
    "memory": {
      "extract_patterns": true,
      "extract_decisions": true,
      "extract_entities": true,
      "consistency_checking": true
    },
    "llm": {
      "temperature": 0.7,
      "model": "claude-3-haiku-20240307"
    }
  }
}
```

All fields are optional. Only the fields you include will be updated.

### Delete Project

```
DELETE /api/v1/projects/{project_id}
```

Delete a project and all associated data.

## Chats

Chats represent conversations with the LLM within the context of a project.

### List Chats

```
GET /api/v1/projects/{project_id}/chats
```

List all chats for a specific project.

### Create Chat

```
POST /api/v1/projects/{project_id}/chats
```

Create a new chat within a project.

**Request Body:**
```json
{
  "name": "Chat Name"
}
```

The `name` field is optional. If not provided, a default name will be generated.

### Get Chat

```
GET /api/v1/chats/{chat_id}
```

Get details for a specific chat.

### List Messages

```
GET /api/v1/chats/{chat_id}/messages
```

List all messages in a chat.

### Send Message

```
POST /api/v1/chats/{chat_id}/messages
```

Send a message to the LLM and get a response.

**Request Body:**
```json
{
  "content": "Your message to the LLM"
}
```

**Response:**
```json
{
  "message": {
    "id": "message_id",
    "chat_id": "chat_id",
    "content": "LLM response",
    "role": "assistant",
    "timestamp": "2023-06-01T12:34:56.789Z",
    "artifacts": [
      {
        "id": "artifact_id",
        "type": "code",
        "language": "python",
        "title": "Example Code",
        "content": "print('Hello, world!')"
      }
    ]
  },
  "consistency_issues": [
    {
      "type": "pattern_conflict",
      "severity": "warning",
      "message": "This pattern conflicts with an established pattern"
    }
  ]
}
```

## Codebase

The codebase endpoints allow you to analyze codebases to extract architectural knowledge.

### Analyze Codebase

```
POST /api/v1/projects/{project_id}/codebase
```

Analyze a codebase to extract architectural knowledge.

**Request Body:**
```json
{
  "codebase_path": "/path/to/codebase",
  "excluded_dirs": "node_modules,venv,.git,__pycache__"
}
```

The `codebase_path` is required unless the project already has an associated codebase path. The `excluded_dirs` field is optional and defaults to a common set of directories to exclude.

**Response:**
```json
{
  "success": true,
  "stats": {
    "files_processed": 153,
    "entities_extracted": 215,
    "relationships_extracted": 342,
    "decisions_extracted": 8,
    "patterns_extracted": 12,
    "languages": {
      "python": 124,
      "javascript": 29
    }
  },
  "message": "Analyzed 153 files successfully"
}
```

## Knowledge

The knowledge endpoints allow you to access the architectural knowledge extracted from codebases and LLM interactions.

### Get Knowledge

```
GET /api/v1/projects/{project_id}/knowledge/{knowledge_type}
```

Get knowledge elements from the knowledge graph. The `knowledge_type` can be one of:
- `decisions`: Architectural decisions
- `patterns`: Design patterns
- `entities`: Code entities (classes, functions, etc.)
- `relationships`: Relationships between entities
- `search`: Semantic search across all knowledge

**Query Parameters:**
- `q`: Search query (required for `search` knowledge type, optional for others)
- `limit`: Maximum number of results to return (default: 1000)
- `offset`: Number of results to skip (for pagination, default: 0)

**Example Response for `decisions`:**
```json
[
  {
    "id": "decision_id",
    "title": "Use Repository Pattern",
    "description": "Decision to use the repository pattern for data access",
    "reasoning": "Provides abstraction over data sources and makes testing easier",
    "alternatives": ["Active Record Pattern", "Direct Database Access"]
  }
]
```

## Consistency Checking

The consistency checking endpoint allows you to check if text or code is consistent with the established architectural knowledge.

### Check Consistency

```
POST /api/v1/projects/{project_id}/consistency
```

Check text for consistency with existing knowledge.

**Request Body:**
```json
{
  "content": "Text or code to check for consistency",
  "artifacts": [
    {
      "type": "code",
      "language": "python",
      "content": "def example(): pass"
    }
  ],
  "store_knowledge": false
}
```

The `artifacts` field is optional. The `store_knowledge` field is optional and defaults to `false`. If set to `true`, any knowledge extracted from the content will be stored in the knowledge graph.

**Response:**
```json
{
  "consistency_issues": [
    {
      "type": "naming_convention",
      "severity": "warning",
      "entity_type": "function",
      "entity_name": "example",
      "expected_pattern": "Functions should use snake_case",
      "message": "The function name 'example' doesn't follow the established naming convention: Functions should use snake_case"
    }
  ],
  "has_issues": true
}
```

## Error Handling

All API endpoints return appropriate HTTP status codes:

- `200 OK`: Request was successful
- `201 Created`: Resource was created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid API key
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error responses include a JSON object with an `error` field containing a description of the error:

```json
{
  "error": "Project not found"
}
```

## Rate Limiting

The API is subject to rate limiting to ensure fair usage. The current limits are:

- 60 requests per minute per API key
- 1000 requests per hour per API key

If you exceed these limits, you'll receive a `429 Too Many Requests` response.

## Examples

### Python Client Example

```python
import requests
import json

API_BASE_URL = "http://localhost:5000/api/v1"
API_KEY = "your_api_key_here"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Create a project
project_data = {
    "name": "My Project",
    "description": "A test project"
}
response = requests.post(f"{API_BASE_URL}/projects", headers=headers, json=project_data)
project = response.json()
project_id = project["id"]

# Create a chat
chat_data = {
    "name": "My Chat"
}
response = requests.post(f"{API_BASE_URL}/projects/{project_id}/chats", headers=headers, json=chat_data)
chat = response.json()
chat_id = chat["id"]

# Send a message
message_data = {
    "content": "Design a simple user authentication system"
}
response = requests.post(f"{API_BASE_URL}/chats/{chat_id}/messages", headers=headers, json=message_data)
result = response.json()

# Check for consistency issues
if result["consistency_issues"]:
    print("Consistency issues found:")
    for issue in result["consistency_issues"]:
        print(f"- {issue['message']}")
else:
    print("No consistency issues found")

# Get architectural decisions
response = requests.get(f"{API_BASE_URL}/projects/{project_id}/knowledge/decisions", headers=headers)
decisions = response.json()
print(f"Found {len(decisions)} architectural decisions")
```

### JavaScript Client Example

```javascript
const API_BASE_URL = 'http://localhost:5000/api/v1';
const API_KEY = 'your_api_key_here';

const headers = {
  'X-API-Key': API_KEY,
  'Content-Type': 'application/json'
};

// Create a project
async function createProject() {
  const projectData = {
    name: 'My Project',
    description: 'A test project'
  };
  
  const response = await fetch(`${API_BASE_URL}/projects`, {
    method: 'POST',
    headers,
    body: JSON.stringify(projectData)
  });
  
  const project = await response.json();
  return project.id;
}

// Create a chat
async function createChat(projectId) {
  const chatData = {
    name: 'My Chat'
  };
  
  const response = await fetch(`${API_BASE_URL}/projects/${projectId}/chats`, {
    method: 'POST',
    headers,
    body: JSON.stringify(chatData)
  });
  
  const chat = await response.json();
  return chat.id;
}

// Send a message
async function sendMessage(chatId, content) {
  const messageData = {
    content
  };
  
  const response = await fetch(`${API_BASE_URL}/chats/${chatId}/messages`, {
    method: 'POST',
    headers,
    body: JSON.stringify(messageData)
  });
  
  return response.json();
}

// Example usage
async function main() {
  try {
    const projectId = await createProject();
    const chatId = await createChat(projectId);
    
    const result = await sendMessage(chatId, 'Design a simple user authentication system');
    
    if (result.consistency_issues && result.consistency_issues.length > 0) {
      console.log('Consistency issues found:');
      result.consistency_issues.forEach(issue => {
        console.log(`- ${issue.message}`);
      });
    } else {
      console.log('No consistency issues found');
    }
    
    // Get architectural decisions
    const decisionsResponse = await fetch(`${API_BASE_URL}/projects/${projectId}/knowledge/decisions`, {
      headers
    });
    
    const decisions = await decisionsResponse.json();
    console.log(`Found ${decisions.length} architectural decisions`);
    
  } catch (error) {
    console.error('Error:', error);
  }
}

main();
```

## API Versioning

This API is versioned. The current version is v1. If breaking changes are introduced in the future, they will be released under a new version (e.g., v2).

## Getting Help

If you encounter any issues or have questions about the API, please [create an issue](https://github.com/yourusername/archivist-llm/issues) on the GitHub repository.
