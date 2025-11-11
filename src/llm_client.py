"""
LLM Client Module

Handles all interactions with Ollama for local LLM inference.
"""

import json
import logging
from typing import Dict, List, Optional, Any
import requests
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for Ollama LLM."""
    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    temperature: float = 0.7
    timeout: int = 120


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.base_url = self.config.base_url
        self.model = self.config.model
        
    def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: Optional[float] = None) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for context
            temperature: Optional temperature override
            
        Returns:
            Generated text response
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(
                url, 
                json=payload, 
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], 
             temperature: Optional[float] = None) -> str:
        """
        Chat completion with conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Optional temperature override
            
        Returns:
            Generated response
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature
            }
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('message', {}).get('content', '').strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in chat completion: {e}")
            raise
    
    def extract_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response and parse it as JSON.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Parsed JSON dictionary
        """
        response = self.generate(prompt, system_prompt, temperature=0.1)
        
        # Try to extract JSON from markdown code blocks first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
        
        # Try to find JSON object in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            response = json_match.group(0)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response was: {response[:500]}")
            # Return empty dict instead of raising
            return {}


import re
