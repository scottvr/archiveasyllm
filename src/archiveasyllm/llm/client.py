"""
Provider-agnostic LLM client that works with different LLM APIs.
"""
from typing import Dict, Any, Tuple, List, Optional
import importlib
import json

class LLMClient:
    """A provider-agnostic client for interacting with LLMs."""
    
    def __init__(self, provider: str, api_key: str, **kwargs):
        """
        Initialize the LLM client.
        
        Args:
            provider: The LLM provider (e.g., "anthropic", "openai")
            api_key: API key for the provider
            **kwargs: Additional provider-specific configuration
        """
        self.provider = provider
        self.api_key = api_key
        self.config = kwargs
        
        # Dynamically load the provider-specific implementation
        try:
            provider_module = importlib.import_module(f"archiveasy.llm.{provider}")
            self.implementation = provider_module.LLMImplementation(api_key, **kwargs)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Unsupported LLM provider: {provider}. Error: {e}")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """
        Create an LLM client from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with provider, api_key, etc.
            
        Returns:
            LLMClient instance
        """
        provider = config.pop("provider")
        api_key = config.pop("api_key")
        return cls(provider, api_key, **config)
    
    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt
            context: Additional context from knowledge graph and vector store
            
        Returns:
            Tuple of (response_text, artifacts)
        """
        # Enhance the prompt with context if provided
        enhanced_prompt = self._enhance_prompt(prompt, context) if context else prompt
        
        # Get a response from the implementation
        response, artifacts = self.implementation.generate(enhanced_prompt)
        
        return response, artifacts
    
    def _enhance_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Enhance the prompt with context from knowledge graph and vector store.
        
        Args:
            prompt: Original user prompt
            context: Dictionary with graph_context and vector_context
            
        Returns:
            Enhanced prompt with added context
        """
        graph_context = context.get("graph_context", [])
        vector_context = context.get("vector_context", [])
        
        # Create a system message with relevant context
        context_str = ""
        
        if graph_context:
            context_str += "## Relevant Design Decisions and Patterns\n\n"
            for item in graph_context:
                context_str += f"- {item['type']}: {item['content']}\n"
            context_str += "\n"
        
        if vector_context:
            context_str += "## Related Previous Discussions\n\n"
            for item in vector_context:
                context_str += f"- {item['content'][:100]}...\n"
            context_str += "\n"
        
        # For Anthropic (Claude), we add the context to the system message
        if self.provider == "anthropic":
            # The actual implementation will be in anthropic.py, this is just a sketch
            return f"""
            <context>
            {context_str}
            </context>
            
            Human: {prompt}
            
            Assistant:
            """
        
        # For OpenAI, we add the context as a system message
        elif self.provider == "openai":
            # The actual implementation will be in openai.py, this is just a sketch
            return json.dumps([
                {"role": "system", "content": context_str},
                {"role": "user", "content": prompt}
            ])
        
        # Default case - just add the context before the prompt
        else:
            return f"{context_str}\n\n{prompt}"
