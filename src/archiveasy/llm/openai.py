# archiveasy/llm/openai.py

import openai
import json
from typing import List, Dict, Tuple

class LLMImplementation:
    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = 0.7, max_tokens: int = 4096, **kwargs):
        openai.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> Tuple[str, List[Dict]]:
        try:
            messages = json.loads(prompt) if prompt.strip().startswith("[") else [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response['choices'][0]['message']['content'].strip(), []
        except Exception as e:
            return f"Error generating response: {e}", []
