"""
Vector database for semantic search of conversation history.
"""
# crumb: memory\vector.py
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np
import os
import faiss
import uuid
import logging
import time

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
        self.dimension = config.get("dimension", 768)  # Default for many embedding models
        self.index_path = config.get("index_path", "./data/vector_indexes")
        self.distance_threshold = config.get("distance_threshold", 0.75)
        
        # Create directory for indexes if it doesn't exist
        os.makedirs(self.index_path, exist_ok=True)
        
        # Dictionary to keep track of project indexes
        self.indexes = {}
        self.metadata = {}
        
        # Check FAISS GPU availability
        try:
            import faiss
            self.gpu_available = hasattr(faiss, 'GpuResourcesProvider')
            if self.gpu_available:
                logger.info("FAISS GPU resources available")
                # Try initializing GPU resources
                try:
                    self.gpu_resources = faiss.StandardGpuResources()
                    logger.info("FAISS GPU resources initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize FAISS GPU resources: {e}")
                    self.gpu_available = False
            else:
                logger.info("FAISS GPU resources not available, using CPU only")
        except Exception as e:
            logger.warning(f"Error checking FAISS GPU availability: {e}")
            self.gpu_available = False
        
        # Load embedding model
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
        
        import time
        start_time = time.time()
        logger.info(f"Loading embedding model: {model_name}")
        
        try:
            # Use sentence-transformers for embeddings
            import torch
            from sentence_transformers import SentenceTransformer
            
            # Log CUDA availability for debugging
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            
            if model_name == "default":
                model_name = "all-MiniLM-L6-v2"  # Small, efficient model
                logger.info(f"Using default model: {model_name}")
            
            # Log start of model loading
            logger.info(f"Starting to load SentenceTransformer model: {model_name}")
            
            # Choose device based on availability
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")
            
            # Load the model with a timeout to detect hangs
            model = SentenceTransformer(model_name, device=device)
            
            load_time = time.time() - start_time
            logger.info(f"Successfully loaded embedding model in {load_time:.2f}s")
            
            # Test the model with a simple example
            logger.info("Testing embedding model with sample text...")
            test_start = time.time()
            sample_embedding = model.encode("This is a test sentence.")
            test_time = time.time() - test_start
            logger.info(f"Test embedding generated in {test_time:.2f}s: shape={sample_embedding.shape}")
            
            return model
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fallback to a very simple embedding method
            logger.warning("Falling back to simple embedding model")
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
        import time
        start_time = time.time()
        
        if project_id not in self.indexes:
            logger.warning(f"No index found for project {project_id}, nothing to save")
            return
        
        index_file = os.path.join(self.index_path, f"{project_id}.index")
        metadata_file = os.path.join(self.index_path, f"{project_id}.json")
        
        logger.info(f"Saving index for project {project_id} to {index_file}")
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(index_file), exist_ok=True)
            
            # Save index
            index_save_start = time.time()
            faiss.write_index(self.indexes[project_id], index_file)
            logger.info(f"FAISS index saved in {time.time() - index_save_start:.2f}s")
            
            # Save metadata
            meta_save_start = time.time()
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata[project_id], f)
            logger.info(f"Metadata saved in {time.time() - meta_save_start:.2f}s")
            
            logger.info(f"Index and metadata saved in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Error saving index or metadata: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue execution despite save error

    def store(self, text: str, artifacts: List[Dict[str, Any]], project_id: str) -> None:
        """
        Store text and artifacts in the vector store.
        
        Args:
            text: Text content to store
            artifacts: List of artifacts
            project_id: Project identifier
        """
        import time
        start_time = time.time()
        
        logger.info(f"Storing content in vector store for project {project_id}")
        
        # Get project index and metadata
        try:
            logger.info("Getting project index")
            index, metadata = self._get_project_index(project_id)
            logger.info(f"Got project index with {index.ntotal if hasattr(index, 'ntotal') else 'unknown'} vectors")
        except Exception as e:
            logger.error(f"Error getting project index: {e}")
            raise
        
        # Store the main text
        try:
            if text:
                logger.info(f"Storing main text ({len(text)} chars)")
                self._store_text(text, "response", project_id, index, metadata)
                logger.info("Main text stored successfully")
            else:
                logger.info("No main text to store")
        except Exception as e:
            logger.error(f"Error storing main text: {e}")
            raise
        
        # Store artifacts
        if artifacts:
            logger.info(f"Storing {len(artifacts)} artifacts")
            for i, artifact in enumerate(artifacts):
                content = artifact.get("content", "")
                artifact_type = artifact.get("type", "unknown")
                
                if content:
                    try:
                        logger.info(f"Storing artifact {i+1}/{len(artifacts)} of type {artifact_type} ({len(content)} chars)")
                        self._store_text(content, f"artifact_{artifact_type}", project_id, index, metadata)
                        logger.info(f"Artifact {i+1} stored successfully")
                    except Exception as e:
                        logger.error(f"Error storing artifact: {e}")
                        continue
        else:
            logger.info("No artifacts to store")
        
        # Save the updated index
        try:
            logger.info("Saving updated vector index")
            self._save_project_index(project_id)
            logger.info("Vector index saved successfully")
        except Exception as e:
            logger.error(f"Error saving vector index: {e}")
            raise
            
        logger.info(f"Vector store operation completed in {time.time() - start_time:.2f}s")

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
        start_time = time.time()
        
        # Split text into chunks for better retrieval
        logger.info(f"Splitting text of length {len(text)} into chunks")
        chunks_start = time.time()
        chunks = self._split_text(text)
        logger.info(f"Split into {len(chunks)} chunks in {time.time() - chunks_start:.2f}s")
        
        # Get embeddings for all chunks
        logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.embedding_model.__class__.__name__}")
        embedding_start = time.time()
        try:
            chunk_embeddings = self.embedding_model.encode(chunks)
            logger.info(f"Generated embeddings in {time.time() - embedding_start:.2f}s, shape: {chunk_embeddings.shape if hasattr(chunk_embeddings, 'shape') else 'unknown'}")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
        
        # Add to index
        logger.info(f"Adding embeddings to FAISS index, current size: {index.ntotal}")
        index_start = time.time()
        start_idx = index.ntotal
        try:
            index.add(chunk_embeddings)
            logger.info(f"Added embeddings to index in {time.time() - index_start:.2f}s, new size: {index.ntotal}")
        except Exception as e:
            logger.error(f"Error adding embeddings to index: {e}")
            raise
        
        # Store metadata for each chunk
        logger.info(f"Storing metadata for {len(chunks)} chunks")
        meta_start = time.time()
        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            metadata[str(idx)] = {
                "text": chunk,
                "source_type": source_type,
                "project_id": project_id,
                "id": str(uuid.uuid4())
            }
        logger.info(f"Metadata stored in {time.time() - meta_start:.2f}s")
        
        logger.info(f"Text storage completed in {time.time() - start_time:.2f}s")

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
