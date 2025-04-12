# archiveasy/llm/anthropic.py

import anthropic
from typing import List, Dict, Tuple

class LLMImplementation:
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307", temperature: float = 0.7, max_tokens: int = 4096, **kwargs):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> Tuple[str, List[Dict]]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text.strip(), []  # Empty artifact list for now
