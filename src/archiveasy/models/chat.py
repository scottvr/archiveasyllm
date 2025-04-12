"""
Chat and Message data models for ArchiveasyLLM.
"""
import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

class Chat:
    """
    Chat data model representing a conversation with the LLM.
    """
    
    data_dir = Path("./data/chats")
    
    def __init__(self, id: str, name: str, project_id: str, 
                 created_at: Optional[datetime] = None,
                 updated_at: Optional[datetime] = None):
        """
        Initialize a chat.
        
        Args:
            id: Unique chat identifier
            name: Chat name
            project_id: Project identifier
            created_at: Creation timestamp
            updated_at: Last update timestamp
        """
        self.id = id
        self.name = name
        self.project_id = project_id
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or self.created_at
    
    @classmethod
    def create(cls, project_id: str, name: Optional[str] = None) -> 'Chat':
        """
        Create a new chat.
        
        Args:
            project_id: Project identifier
            name: Chat name (optional)
            
        Returns:
            New Chat instance
        """
        # Generate unique ID
        chat_id = str(uuid.uuid4())
        
        # Generate name if not provided
        if name is None:
            name = f"Chat {cls.count_by_project(project_id) + 1}"
        
        # Create chat directory
        chat_dir = cls.data_dir / chat_id
        os.makedirs(chat_dir, exist_ok=True)
        
        # Create messages directory
        messages_dir = chat_dir / "messages"
        os.makedirs(messages_dir, exist_ok=True)
        
        # Create chat instance
        now = datetime.now()
        chat = cls(
            id=chat_id,
            name=name,
            project_id=project_id,
            created_at=now,
            updated_at=now
        )
        
        # Save to disk
        chat._save()
        
        return chat
    
    @classmethod
    def get(cls, chat_id: str) -> Optional['Chat']:
        """
        Get a chat by ID.
        
        Args:
            chat_id: Chat identifier
            
        Returns:
            Chat instance or None if not found
        """
        metadata_file = cls.data_dir / chat_id / "metadata.json"
        
        if not os.path.exists(metadata_file):
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            # Parse timestamps
            created_at = datetime.fromisoformat(data.get('created_at')) if 'created_at' in data else None
            updated_at = datetime.fromisoformat(data.get('updated_at')) if 'updated_at' in data else None
            
            return cls(
                id=data.get('id'),
                name=data.get('name'),
                project_id=data.get('project_id'),
                created_at=created_at,
                updated_at=updated_at
            )
        except Exception as e:
            print(f"Error loading chat {chat_id}: {e}")
            return None
    
    @classmethod
    def list_by_project(cls, project_id: str) -> List['Chat']:
        """
        List all chats for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of Chat instances
        """
        chats = []
        
        if not os.path.exists(cls.data_dir):
            os.makedirs(cls.data_dir, exist_ok=True)
            return chats
        
        for chat_id in os.listdir(cls.data_dir):
            chat_dir = cls.data_dir / chat_id
            
            if not os.path.isdir(chat_dir):
                continue
                
            metadata_file = chat_dir / "metadata.json"
            
            if not os.path.exists(metadata_file):
                continue
                
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                if data.get('project_id') == project_id:
                    # Parse timestamps
                    created_at = datetime.fromisoformat(data.get('created_at')) if 'created_at' in data else None
                    updated_at = datetime.fromisoformat(data.get('updated_at')) if 'updated_at' in data else None
                    
                    chat = cls(
                        id=data.get('id'),
                        name=data.get('name'),
                        project_id=data.get('project_id'),
                        created_at=created_at,
                        updated_at=updated_at
                    )
                    chats.append(chat)
            except Exception as e:
                print(f"Error loading chat metadata: {e}")
        
        # Sort by updated_at (newest first)
        chats.sort(key=lambda c: c.updated_at, reverse=True)
        
        return chats
    
    @classmethod
    def count_by_project(cls, project_id: str) -> int:
        """
        Count chats for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Number of chats
        """
        return len(cls.list_by_project(project_id))
    
    def update(self, **kwargs) -> 'Chat':
        """
        Update chat attributes.
        
        Args:
            **kwargs: Attributes to update
            
        Returns:
            Updated Chat instance
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Update updated_at timestamp
        self.updated_at = datetime.now()
        
        # Save changes
        self._save()
        
        return self
    
    def delete(self) -> bool:
        """
        Delete the chat.
        
        Returns:
            True if successful
        """
        import shutil
        
        chat_dir = self.data_dir / self.id
        
        if os.path.exists(chat_dir):
            try:
                shutil.rmtree(chat_dir)
                return True
            except Exception as e:
                print(f"Error deleting chat {self.id}: {e}")
        
        return False
    
    def _save(self) -> None:
        """Save chat metadata to disk."""
        chat_dir = self.data_dir / self.id
        os.makedirs(chat_dir, exist_ok=True)
        
        metadata_file = chat_dir / "metadata.json"
        
        data = {
            'id': self.id,
            'name': self.name,
            'project_id': self.project_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary of chat attributes
        """
        return {
            'id': self.id,
            'name': self.name,
            'project_id': self.project_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'message_count': Message.count_by_chat(self.id)
        }


class Message:
    """
    Message data model representing a single message in a chat.
    """
    
    def __init__(self, id: str, chat_id: str, content: str, role: str, 
                 timestamp: Optional[datetime] = None,
                 artifacts: Optional[List[Dict[str, Any]]] = None,
                 consistency_issues: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize a message.
        
        Args:
            id: Unique message identifier
            chat_id: Chat identifier
            content: Message content
            role: Message role (user, assistant, system)
            timestamp: Creation timestamp
            artifacts: List of artifacts (code, markdown, etc.)
            consistency_issues: List of consistency issues
        """
        self.id = id
        self.chat_id = chat_id
        self.content = content
        self.role = role
        self.timestamp = timestamp or datetime.now()
        self.artifacts = artifacts or []
        self.consistency_issues = consistency_issues or []
    
    @classmethod
    def create(cls, chat_id: str, content: str, role: str, 
              artifacts: Optional[List[Dict[str, Any]]] = None,
              consistency_issues: Optional[List[Dict[str, Any]]] = None) -> 'Message':
        """
        Create a new message.
        
        Args:
            chat_id: Chat identifier
            content: Message content
            role: Message role (user, assistant, system)
            artifacts: List of artifacts (code, markdown, etc.)
            consistency_issues: List of consistency issues
            
        Returns:
            New Message instance
        """
        # Generate unique ID
        message_id = str(uuid.uuid4())
        
        # Create message instance
        message = cls(
            id=message_id,
            chat_id=chat_id,
            content=content,
            role=role,
            timestamp=datetime.now(),
            artifacts=artifacts,
            consistency_issues=consistency_issues
        )
        
        # Save to disk
        message._save()
        
        # Update chat's updated_at timestamp
        chat = Chat.get(chat_id)
        if chat:
            chat.update()
        
        return message
    
    @classmethod
    def get(cls, message_id: str, chat_id: str) -> Optional['Message']:
        """
        Get a message by ID.
        
        Args:
            message_id: Message identifier
            chat_id: Chat identifier
            
        Returns:
            Message instance or None if not found
        """
        message_file = Chat.data_dir / chat_id / "messages" / f"{message_id}.json"
        
        if not os.path.exists(message_file):
            return None
        
        try:
            with open(message_file, 'r') as f:
                data = json.load(f)
            
            # Parse timestamp
            timestamp = datetime.fromisoformat(data.get('timestamp')) if 'timestamp' in data else None
            
            return cls(
                id=data.get('id'),
                chat_id=data.get('chat_id'),
                content=data.get('content'),
                role=data.get('role'),
                timestamp=timestamp,
                artifacts=data.get('artifacts', []),
                consistency_issues=data.get('consistency_issues', [])
            )
        except Exception as e:
            print(f"Error loading message {message_id}: {e}")
            return None
    
    @classmethod
    def list_by_chat(cls, chat_id: str) -> List['Message']:
        """
        List all messages for a chat.
        
        Args:
            chat_id: Chat identifier
            
        Returns:
            List of Message instances
        """
        messages = []
        
        messages_dir = Chat.data_dir / chat_id / "messages"
        
        if not os.path.exists(messages_dir):
            os.makedirs(messages_dir, exist_ok=True)
            return messages
        
        for filename in os.listdir(messages_dir):
            if not filename.endswith('.json'):
                continue
                
            message_file = messages_dir / filename
            
            try:
                with open(message_file, 'r') as f:
                    data = json.load(f)
                
                # Parse timestamp
                timestamp = datetime.fromisoformat(data.get('timestamp')) if 'timestamp' in data else None
                
                message = cls(
                    id=data.get('id'),
                    chat_id=data.get('chat_id'),
                    content=data.get('content'),
                    role=data.get('role'),
                    timestamp=timestamp,
                    artifacts=data.get('artifacts', []),
                    consistency_issues=data.get('consistency_issues', [])
                )
                messages.append(message)
            except Exception as e:
                print(f"Error loading message {filename}: {e}")
        
        # Sort by timestamp
        messages.sort(key=lambda m: m.timestamp)
        
        return messages
    
    @classmethod
    def count_by_chat(cls, chat_id: str) -> int:
        """
        Count messages for a chat.
        
        Args:
            chat_id: Chat identifier
            
        Returns:
            Number of messages
        """
        messages_dir = Chat.data_dir / chat_id / "messages"
        
        if not os.path.exists(messages_dir):
            return 0
        
        return len([f for f in os.listdir(messages_dir) if f.endswith('.json')])
    
    def update(self, **kwargs) -> 'Message':
        """
        Update message attributes.
        
        Args:
            **kwargs: Attributes to update
            
        Returns:
            Updated Message instance
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Save changes
        self._save()
        
        return self
    
    def delete(self) -> bool:
        """
        Delete the message.
        
        Returns:
            True if successful
        """
        message_file = Chat.data_dir / self.chat_id / "messages" / f"{self.id}.json"
        
        if os.path.exists(message_file):
            try:
                os.remove(message_file)
                return True
            except Exception as e:
                print(f"Error deleting message {self.id}: {e}")
        
        return False
    
    def _save(self) -> None:
        """Save message to disk."""
        messages_dir = Chat.data_dir / self.chat_id / "messages"
        os.makedirs(messages_dir, exist_ok=True)
        
        message_file = messages_dir / f"{self.id}.json"
        
        data = {
            'id': self.id,
            'chat_id': self.chat_id,
            'content': self.content,
            'role': self.role,
            'timestamp': self.timestamp.isoformat(),
            'artifacts': self.artifacts,
            'consistency_issues': self.consistency_issues
        }
        
        with open(message_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary of message attributes
        """
        return {
            'id': self.id,
            'chat_id': self.chat_id,
            'content': self.content,
            'role': self.role,
            'timestamp': self.timestamp.isoformat(),
            'artifacts': self.artifacts,
            'consistency_issues': self.consistency_issues
        }
