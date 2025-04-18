"""
Ollama integration for local LLM inference
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from core.engine.logging import debug

class OllamaConnector:
    """Provides integration with Ollama for local LLM inference"""
    
    def __init__(self, config):
        """Initialize Ollama connector with configuration"""
        self.config = config
        self.host = config.get('llm.ollama_host')
        self.port = config.get('llm.ollama_port')
        self.base_url = f"{self.host}:{self.port}"
        self.available_models = None
        debug(config, f"Ollama connector initialized with base URL: {self.base_url}")
        
    def check_connection(self) -> bool:
        """
        Check connection to Ollama server
        
        Returns:
            bool: Connection status
        """
        debug(self.config, "Checking connection to Ollama server")
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.available_models = response.json().get('models', [])
                model_names = [model.get('name') for model in self.available_models]
                debug(self.config, f"Connected to Ollama. Available models: {model_names}")
                return True
            else:
                debug(self.config, f"Failed to connect to Ollama. Status code: {response.status_code}")
                return False
        except Exception as e:
            debug(self.config, f"Error connecting to Ollama: {str(e)}")
            return False
            
    def list_models(self) -> List[str]:
        """
        List available models from Ollama
        
        Returns:
            List[str]: List of available model names
        """
        if not self.available_models:
            if not self.check_connection():
                return []
                
        return [model.get('name') for model in self.available_models]
        
    def generate(self, prompt: str, model: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """
        Generate text using Ollama
        
        Args:
            prompt (str): Input prompt
            model (str, optional): Model name. If None, uses default from config.
            temperature (float): Sampling temperature (0.0 to 1.0)
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            str: Generated text
        """
        debug(self.config, f"Generating text with Ollama")
        
        # Use default model from config if not specified
        if model is None:
            model = self.config.get('llm.default_model')
        
        # Check if model is available
        if not self.check_model_availability(model):
            fallback_model = "mistral"  # Safe fallback
            debug(self.config, f"Model {model} not available. Falling back to {fallback_model}")
            model = fallback_model
            
        try:
            # Prepare request payload
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            # Make API call
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=60
            )
            
            # Process response
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                debug(self.config, f"Error from Ollama API: {response.status_code}")
                return f"Error generating text: HTTP {response.status_code}"
                
        except Exception as e:
            debug(self.config, f"Error generating text with Ollama: {str(e)}")
            return f"Error generating text: {str(e)}"
            
    def generate_with_context(self, query: str, context_docs: List[Dict[str, Any]], 
                             model: Optional[str] = None, temperature: float = 0.7) -> str:
        """
        Generate text with document context
        
        Args:
            query (str): User query
            context_docs (List[Dict]): Context documents
            model (str, optional): Model name. If None, uses default from config.
            temperature (float): Sampling temperature
            
        Returns:
            str: Generated response
        """
        debug(self.config, f"Generating text with context using Ollama")
        
        # Build prompt with context
        prompt = "You are a document analysis assistant. Answer the query based on the provided context documents.\n\n"
        
        # Add context documents
        prompt += "Context documents:\n"
        for i, doc in enumerate(context_docs):
            prompt += f"\n--- Document {i+1} ---\n"
            prompt += doc.get("content", "")[:1000]  # Limit context size per document
            prompt += "\n"
        
        # Add the query
        prompt += f"\n\nQuery: {query}\n\nAnswer: "
        
        # Generate response
        return self.generate(prompt, model, temperature, max_tokens=1024)
        
    def check_model_availability(self, model_name: str) -> bool:
        """
        Check if a model is available
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            bool: Whether the model is available
        """
        if not self.available_models:
            if not self.check_connection():
                return False
                
        return any(model.get('name') == model_name for model in self.available_models)
