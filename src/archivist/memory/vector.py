"""
Vector database for semantic search of conversation history.
"""
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np
import os
import faiss
import uuid
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector store for semantic search of conversation history and knowledge.
    Uses FAISS for efficient vector storage and retrieval.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vector store.
        
        Args:
            config: Configuration for the vector store
                embedding_model: Model for text embeddings
                dimension: Embedding dimension
                index_path: Path to save FAISS indexes
                distance_threshold: Maximum distance for relevant results
        """
        self.config = config
        self.embedding_model = self._load_embedding_model(config.get("embedding_model", "default"))
        self.dimension = config.get("dimension", 768)  # Default for many embedding models
        self.index_path = config.get("index_path", "./data/vector_indexes")
        self.distance_threshold = config.get("distance_threshold", 0.75)
        
        # Create directory for indexes if it doesn't exist
        os.makedirs(self.index_path, exist_ok=True)
        
        # Dictionary to keep track of project indexes
        self.indexes = {}
        self.metadata = {}
    
    def _load_embedding_model(self, model_name: str):
        """
        Load the embedding model.
        
        Args:
            model_name: Name of the embedding model to load
            
        Returns:
            Embedding model
        """
        # In a real implementation, this would load the appropriate model
        # For simplicity, we'll implement a basic embedding function
        
        from sentence_transformers import SentenceTransformer
        
        try:
            # Use sentence-transformers for embeddings
            if model_name == "default":
                model_name = "all-MiniLM-L6-v2"  # Small, efficient model
            
            return SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            # Fallback to a very simple embedding method
            return self._fallback_embedding_model()
    
    def _fallback_embedding_model(self):
        """
        Fallback embedding model in case the main one fails to load.
        
        Returns:
            A simple callable that returns random vectors (for testing only)
        """
        logger.warning("Using fallback embedding model - random vectors only!")
        
        class FallbackModel:
            def encode(self, texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                return np.random.randn(len(texts), 768).astype(np.float32)
        
        return FallbackModel()
    
    def _get_project_index(self, project_id: str) -> Tuple[faiss.Index, Dict[int, Dict[str, Any]]]:
        """
        Get or create FAISS index for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Tuple of (FAISS index, metadata dictionary)
        """
        if project_id in self.indexes:
            return self.indexes[project_id], self.metadata[project_id]
        
        # Check if index file exists
        index_file = os.path.join(self.index_path, f"{project_id}.index")
        metadata_file = os.path.join(self.index_path, f"{project_id}.json")
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            # Load existing index and metadata
            index = faiss.read_index(index_file)
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            # Create new index and metadata
            index = faiss.IndexFlatL2(self.dimension)
            metadata = {}
        
        self.indexes[project_id] = index
        self.metadata[project_id] = metadata
        
        return index, metadata
    
    def _save_project_index(self, project_id: str):
        """
        Save FAISS index and metadata for a project.
        
        Args:
            project_id: Project identifier
        """
        if project_id not in self.indexes:
            return
        
        index_file = os.path.join(self.index_path, f"{project_id}.index")
        metadata_file = os.path.join(self.index_path, f"{project_id}.json")
        
        # Save index
        faiss.write_index(self.indexes[project_id], index_file)
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata[project_id], f)
    
    def store(self, text: str, artifacts: List[Dict[str, Any]], project_id: str) -> None:
        """
        Store text and artifacts in the vector store.
        
        Args:
            text: Text content to store
            artifacts: List of artifacts
            project_id: Project identifier
        """
        # Get project index and metadata
        index, metadata = self._get_project_index(project_id)
        
        # Store the main text
        self._store_text(text, "response", project_id, index, metadata)
        
        # Store artifacts
        for artifact in artifacts:
            content = artifact.get("content", "")
            artifact_type = artifact.get("type", "unknown")
            
            if content:
                self._store_text(content, f"artifact_{artifact_type}", project_id, index, metadata)
        
        # Save the updated index
        self._save_project_index(project_id)
    
    def _store_text(self, text: str, source_type: str, project_id: str, 
                   index: faiss.Index, metadata: Dict[int, Dict[str, Any]]) -> None:
        """
        Store text in the vector store.
        
        Args:
            text: Text content to store
            source_type: Type of source (response, artifact, etc.)
            project_id: Project identifier
            index: FAISS index
            metadata: Metadata dictionary
        """
        # Split text into chunks for better retrieval
        chunks = self._split_text(text)
        
        # Get embeddings for all chunks
        chunk_embeddings = self.embedding_model.encode(chunks)
        
        # Add to index
        start_idx = index.ntotal
        index.add(chunk_embeddings)
        
        # Store metadata for each chunk
        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            metadata[str(idx)] = {
                "text": chunk,
                "source_type": source_type,
                "project_id": project_id,
                "id": str(uuid.uuid4())
            }
    
    def _split_text(self, text: str, max_chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval.
        
        Args:
            text: Text to split
            max_chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk end position
            end = start + max_chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to find a sentence break for cleaner chunks
            sentence_break = text.rfind('. ', start, end) + 1
            
            if sentence_break > start:
                end = sentence_break
            
            chunks.append(text[start:end])
            
            # Move start position with overlap
            start = end - overlap
        
        return chunks
    
    def search(self, query: str, project_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant content using semantic similarity.
        
        Args:
            query: Search query
            project_id: Project identifier
            limit: Maximum number of results
            
        Returns:
            List of relevant items
        """
        # Get project index and metadata
        try:
            index, metadata = self._get_project_index(project_id)
        except Exception as e:
            logger.error(f"Error accessing vector index: {e}")
            return []
        
        # Check if index is empty
        if index.ntotal == 0:
            return []
        
        # Get query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search index
        distances, indices = index.search(query_embedding, min(limit, index.ntotal))
        
        results = []
        seen_content = set()  # To avoid duplicate content
        
        # Process results
        for i, idx in enumerate(indices[0]):
            # Skip if distance is too large (not relevant)
            if distances[0][i] > self.distance_threshold:
                continue
                
            # Get metadata for this index
            meta = metadata.get(str(idx), {})
            content = meta.get("text", "")
            
            # Skip if we've seen this content already
            content_hash = hash(content)
            if content_hash in seen_content:
                continue
                
            seen_content.add(content_hash)
            
            # Add to results
            results.append({
                "id": meta.get("id", str(uuid.uuid4())),
                "content": content,
                "source_type": meta.get("source_type", "unknown"),
                "relevance": float(1.0 - distances[0][i] / self.distance_threshold)
            })
        
        return results
