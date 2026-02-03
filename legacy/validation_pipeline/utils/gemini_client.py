"""
Google Gemini API Client for Validation Pipeline
"""

import os
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
try:
    # Optional dependency; validation can still run if env vars are set elsewhere.
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None

# Try new API first, fallback to old API
try:
    import google.genai as genai
    USE_NEW_API = True
except ImportError:
    try:
        # Fallback to old API for backward compatibility
        import google.generativeai as genai
        USE_NEW_API = False
    except ImportError:
        raise ImportError(
            "Google Gemini SDK not installed. Install with: pip install google-genai"
        )

# Load environment variables
if load_dotenv is not None:
    load_dotenv()


class GeminiClient:
    """Client for interacting with Google Gemini API"""
    
    def __init__(self, model: str = "models/gemini-3-flash-preview", api_key: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            model: Model name (e.g., "models/gemini-3-flash-preview")
            api_key: API key (if None, reads from GOOGLE_API_KEY env var)
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please set it in .env file or as environment variable."
            )
        
        self.api_key = api_key
        self.model = model
        self._uploaded_files: Dict[str, Any] = {}  # Cache for uploaded files
        
        # Initialize based on API version
        if USE_NEW_API:
            # New API: use Client
            try:
                self.client = genai.Client(api_key=api_key)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Gemini Client: {e}")
        else:
            # Old API: use configure
            try:
                genai.configure(api_key=api_key)
                self.client = None  # Old API doesn't use client
            except Exception as e:
                raise RuntimeError(f"Failed to configure Gemini API: {e}")
    
    def upload_file(self, file_path: Path) -> Any:
        """
        Upload a file (PDF, image, etc.) to Gemini API and wait for it to be ACTIVE.
        
        Args:
            file_path: Path to file to upload
            
        Returns:
            Uploaded file object
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check cache
        cache_key = str(file_path.absolute())
        if cache_key in self._uploaded_files:
            return self._uploaded_files[cache_key]
        
        # Upload file
        try:
            if USE_NEW_API:
                # New API: use client.files.upload(file=...)
                uploaded_file = self.client.files.upload(file=str(file_path))
                
                # Wait for file to be active
                import time
                while uploaded_file.state.name == "PROCESSING":
                    time.sleep(2)
                    uploaded_file = self.client.files.get(name=uploaded_file.name)
                
                if uploaded_file.state.name != "ACTIVE":
                    raise RuntimeError(f"File {file_path.name} failed to process: {uploaded_file.state.name}")
            else:
                # Old API: use genai.upload_file()
                uploaded_file = genai.upload_file(path=str(file_path))
                
                # Wait for file to be active
                import time
                while uploaded_file.state.name == "PROCESSING":
                    time.sleep(2)
                    uploaded_file = genai.get_file(uploaded_file.name)
                
                if uploaded_file.state.name != "ACTIVE":
                    raise RuntimeError(f"File {file_path.name} failed to process: {uploaded_file.state.name}")
            
            self._uploaded_files[cache_key] = uploaded_file
            return uploaded_file
        except Exception as e:
            raise RuntimeError(f"Error uploading file {file_path}: {e}")
    
    def _generate_content_with_model(
        self,
        model_name: str,
        prompt: Union[str, List[Union[str, Any]]],
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Internal method to generate content with a specific model.
        
        Args:
            model_name: Model name to use
            prompt: User prompt (str) or list of content parts (can include uploaded files)
            system_instruction: Optional system instruction
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If RECITATION finish_reason is detected
        """
        try:
            if USE_NEW_API:
                # New API: use client.models.generate_content()
                # Build generation config
                config_dict = {}
                if temperature is not None:
                    config_dict['temperature'] = temperature
                if max_tokens is not None:
                    config_dict['max_output_tokens'] = max_tokens
                if system_instruction is not None:
                    config_dict['system_instruction'] = system_instruction
                
                # Disable all safety filters
                try:
                    if hasattr(genai.types, 'SafetySetting'):
                        safety_settings = [
                            genai.types.SafetySetting(
                                category=genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            genai.types.SafetySetting(
                                category=genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            genai.types.SafetySetting(
                                category=genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            genai.types.SafetySetting(
                                category=genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                        ]
                        config_dict['safety_settings'] = safety_settings
                except (AttributeError, TypeError) as e:
                    # If safety settings not available, continue without them
                    pass
                
                config = genai.types.GenerateContentConfig(**config_dict) if config_dict else None
                
                # Prepare content
                contents = prompt
                
                # Generate content
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                
                # Check for finish_reason before accessing text
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = candidate.finish_reason
                        if finish_reason and finish_reason.name == 'RECITATION':
                            error_msg = f"Response has no text content. Finish reason: RECITATION"
                            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                                safety_ratings = candidate.safety_ratings
                                error_msg += f" Safety ratings: {safety_ratings}"
                            else:
                                error_msg += " Safety ratings: None"
                            error_msg += " This may indicate the content was blocked or filtered."
                            raise RuntimeError(error_msg)
                
                # Try to access text, with fallback error handling
                try:
                    return response.text
                except (AttributeError, ValueError) as e:
                    # Additional check if text access fails
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        finish_reason_str = "unknown"
                        if hasattr(candidate, 'finish_reason'):
                            finish_reason = candidate.finish_reason
                            finish_reason_str = finish_reason.name if hasattr(finish_reason, 'name') else str(finish_reason)
                        error_msg = f"Response has no text content. Finish reason: {finish_reason_str}"
                        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                            safety_ratings = candidate.safety_ratings
                            error_msg += f" Safety ratings: {safety_ratings}"
                        else:
                            error_msg += " Safety ratings: None"
                        error_msg += " This may indicate the content was blocked or filtered."
                        raise RuntimeError(error_msg)
                    raise
            else:
                # Old API: use GenerativeModel
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                )
                if max_tokens:
                    generation_config.max_output_tokens = max_tokens
                
                # Disable all safety filters for old API
                safety_settings = [
                    {
                        "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                    },
                ]
                
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction,
                    safety_settings=safety_settings,
                )
                
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                
                # Check for finish_reason before accessing text (old API)
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = candidate.finish_reason
                        # Handle both enum and string finish_reasons
                        finish_reason_str = finish_reason.name if hasattr(finish_reason, 'name') else str(finish_reason)
                        if finish_reason_str == 'RECITATION' or (isinstance(finish_reason, int) and finish_reason == 4):
                            error_msg = f"Response has no text content. Finish reason: RECITATION"
                            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                                safety_ratings = candidate.safety_ratings
                                error_msg += f" Safety ratings: {safety_ratings}"
                            else:
                                error_msg += " Safety ratings: None"
                            error_msg += " This may indicate the content was blocked or filtered."
                            raise RuntimeError(error_msg)
                
                # Try to access text, with fallback error handling
                try:
                    return response.text
                except (AttributeError, ValueError) as e:
                    # Additional check if text access fails
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        finish_reason_str = "unknown"
                        if hasattr(candidate, 'finish_reason'):
                            finish_reason = candidate.finish_reason
                            finish_reason_str = finish_reason.name if hasattr(finish_reason, 'name') else str(finish_reason)
                        error_msg = f"Response has no text content. Finish reason: {finish_reason_str}"
                        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                            safety_ratings = candidate.safety_ratings
                            error_msg += f" Safety ratings: {safety_ratings}"
                        else:
                            error_msg += " Safety ratings: None"
                        error_msg += " This may indicate the content was blocked or filtered."
                        raise RuntimeError(error_msg)
                    raise
        except Exception as e:
            raise RuntimeError(f"Error generating content with Gemini API: {e}")
    
    def generate_content(
        self,
        prompt: Union[str, List[Union[str, Any]]],
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate content using Gemini API with automatic fallback on RECITATION errors.
        
        Args:
            prompt: User prompt (str) or list of content parts (can include uploaded files)
            system_instruction: Optional system instruction
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        # Fallback models to try if RECITATION error occurs
        fallback_models = [
            self.model,  # First try: configured model
            "models/gemini-flash-latest",  # Second try: flash latest
            "models/gemini-flash-latest",  # Third try: flash latest again
        ]
        
        last_error = None
        for attempt, model_name in enumerate(fallback_models, 1):
            try:
                return self._generate_content_with_model(
                    model_name=model_name,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except RuntimeError as e:
                error_msg = str(e)
                # Check if this is a RECITATION error (check multiple patterns)
                is_recitation_error = (
                    "RECITATION" in error_msg.upper() or 
                    "Finish reason: RECITATION" in error_msg or
                    "finish_reason" in error_msg.lower() and "RECITATION" in error_msg.upper()
                )
                
                if is_recitation_error:
                    last_error = e
                    if attempt < len(fallback_models):
                        # Try next fallback model
                        print(f"⚠️  RECITATION error with {model_name}, trying fallback model {fallback_models[attempt]}...")
                        continue
                    else:
                        # All models failed with RECITATION
                        raise RuntimeError(
                            f"All fallback models failed with RECITATION error. "
                            f"Tried: {', '.join(fallback_models)}. "
                            f"Last error: {error_msg}"
                        )
                else:
                    # Non-RECITATION error, re-raise immediately
                    raise
        
        # If we get here, all attempts failed
        if last_error:
            raise last_error
        raise RuntimeError("Failed to generate content with all fallback models")
    
    def generate_structured(
        self,
        prompt: Union[str, List[Union[str, Any]]],
        system_instruction: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Generate structured response (JSON).
        
        Args:
            prompt: User prompt (str) or list of content parts (can include uploaded files)
            system_instruction: Optional system instruction
            response_format: Response format (default: "json")
            temperature: Sampling temperature (lower for more deterministic)
            
        Returns:
            Parsed JSON response as dictionary
        """
        import json
        
        format_instruction = ""
        if response_format == "json":
            format_instruction = "\n\nPlease respond in valid JSON format only, without markdown code blocks."
        
        # Handle both string and list prompts
        if isinstance(prompt, str):
            full_prompt = f"{prompt}{format_instruction}"
        else:
            # For list prompts, append format instruction as text
            full_prompt = prompt + [format_instruction]
        
        response_text = self.generate_content(
            prompt=full_prompt,
            system_instruction=system_instruction,
            temperature=temperature,
        )
        
        # Try to extract JSON from response
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON response: {e}\n"
                f"Response text: {response_text[:500]}"
            )
