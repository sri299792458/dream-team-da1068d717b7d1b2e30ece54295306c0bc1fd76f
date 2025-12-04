"""
LLM interface module for Dream Team framework.

Provides a wrapper for Gemini API (google-genai) with caching and error handling.
"""

from google import genai
from google.genai import types
from typing import List, Dict, Optional
import os
import time
import json
import re


class GeminiLLM:
    """Wrapper for Gemini API with caching and error handling"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-pro",
        temperature: float = 0.7,
        max_retries: int = 3,
#         thinking_level: str = "low"  # "low" or "high"
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
#         self.thinking_level = thinking_level

        # Simple stats
        self.total_calls = 0
        self.total_tokens = 0

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
#         thinking_level: Optional[str] = None,
        response_format: str = "text"  # "text" or "json"
    ) -> str:
        """Generate response with retry logic"""

        temp = temperature if temperature is not None else self.temperature
#         think = thinking_level if thinking_level is not None else self.thinking_level

        # Map string thinking level to enum/string expected by SDK
        # "low" -> "THINKING_LEVEL_LOW" or just "LOW" usually works, but let's be safe with string
        # The new SDK often accepts strings for enums.
        # Based on research, it's "LOW" or "HIGH" (case insensitive usually)
        
        # Ensure thinking level is valid
#         if think.lower() not in ["low", "high"]:
#             think = "low" # Default fallback

        for attempt in range(self.max_retries):
            try:
                # Configure generation
                config_args = {
                    "temperature": temp,
#                     "thinking_config": types.ThinkingConfig(
#                         include_thoughts=False, # We usually don't want thoughts in output unless requested
#                         thinking_level=think.upper()
#                     )
                }

                if response_format == "json":
                    config_args["response_mime_type"] = "application/json"
                
                if system_instruction:
                    config_args["system_instruction"] = system_instruction

                config = types.GenerateContentConfig(**config_args)

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )

                self.total_calls += 1
                
                return response.text

            except Exception as e:
                print(f"⚠️  Gemini API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
#         thinking_level: Optional[str] = None
    ) -> Dict:
        """Generate JSON response"""
        response_text = self.generate(
            prompt,
            system_instruction=system_instruction,
            temperature=temperature,
#             thinking_level=thinking_level,
            response_format="json"
        )

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from markdown code blocks
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
#         thinking_level: Optional[str] = None
    ) -> str:
        """Multi-turn chat (for meetings)"""
        temp = temperature if temperature is not None else self.temperature
#         think = thinking_level if thinking_level is not None else self.thinking_level
        
#         if think.lower() not in ["low", "high"]:
#             think = "low"

        # Convert to Gemini format
        # messages format: [{"role": "user"/"assistant", "content": "..."}]
        
        # The new SDK chat interface is slightly different.
        # It's better to use generate_content with a list of contents for stateless, 
        # or maintain a chat session.
        # Given the usage pattern, we are often passing a full history.
        
        # Convert history
        gemini_contents = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])]
            ))
            
        # Create chat session
        chat = self.client.chats.create(
            model=self.model_name,
            history=gemini_contents,
            config=types.GenerateContentConfig(
                temperature=temp,
#                 thinking_config=types.ThinkingConfig(
#                     include_thoughts=False,
#                     thinking_level=think.upper()
#                 )
            )
        )

        # Send last message
        response = chat.send_message(messages[-1]["content"])

        self.total_calls += 1
        return response.text


# Global instance (lazy init)
_llm_instance = None


def get_llm(**kwargs) -> GeminiLLM:
    """Get or create global LLM instance"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = GeminiLLM(**kwargs)
    return _llm_instance
