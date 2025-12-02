"""
LLM Handler for Google Gemini API integration with multilingual support.
Handles communication with Gemini API and multilingual prompt construction.
"""

import os
import json
import re
import time
from typing import List, Dict, Any, Optional
from utils import get_language_name

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class GeminiHandler:
    """
    Handler for Google Gemini API integration with multilingual document QA capabilities.
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Gemini Handler.
        
        Args:
            api_key: Gemini API key (default: from environment)
            model: Gemini model to use (default: gemini-pro)
        """
        if genai is None:
            raise ImportError(
                "google-generativeai package is required. Install it with: pip install google-generativeai"
            )
        
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = model or os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Please set GEMINI_API_KEY environment variable or create a .env file. "
                "Get your API key from: https://makersuite.google.com/app/apikey"
            )
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        
        # Validate model exists before creating client
        try:
            available_models = self.get_available_models()
            if self.model not in available_models:
                # Try to use default if current model not available
                default_model = "gemini-2.5-flash"
                if default_model in available_models:
                    print(f"Warning: Model '{self.model}' not available. Using '{default_model}' instead.")
                    self.model = default_model
                else:
                    # Use first available model
                    if available_models:
                        print(f"Warning: Model '{self.model}' not available. Using '{available_models[0]}' instead.")
                        self.model = available_models[0]
                    else:
                        raise ValueError(f"Model '{self.model}' is not available and no fallback models found.")
        except Exception as e:
            print(f"Warning: Could not validate model: {e}")
        
        self.client = genai.GenerativeModel(self.model)
    
    def _optimize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Optimize messages to reduce payload size and avoid connection issues.
        
        Args:
            messages: Original messages
            
        Returns:
            Optimized messages with truncated context if needed
        """
        optimized = messages.copy()
        total_size = sum(len(json.dumps(msg)) for msg in optimized)
        
        # If total size is too large, truncate context in user message
        if total_size > 30000:  # ~30KB limit
            print(f"⚠️ Optimizing messages (size: {total_size/1024:.1f}KB)")
            
            if len(optimized) > 1 and 'content' in optimized[-1]:
                user_content = optimized[-1]['content']
                
                # If context is present, truncate it
                if 'DOCUMENT CONTEXT:' in user_content:
                    # Split into parts
                    context_part = user_content.split('DOCUMENT CONTEXT:')[1].split('USER QUESTION:')[0] if 'USER QUESTION:' in user_content else ''
                    query_part = user_content.split('USER QUESTION:')[1] if 'USER QUESTION:' in user_content else user_content
                    
                    # Truncate context to ~15000 chars
                    if len(context_part) > 15000:
                        context_part = context_part[:15000] + "\n\n[Context truncated for performance]"
                    
                    # Reconstruct
                    optimized[-1]['content'] = f"DOCUMENT CONTEXT:\n{context_part}\n\nUSER QUESTION: {query_part}"
        
        return optimized
    
    TRANSLATABLE_LANGUAGES = {
        "English": "English",
        "Malayalam": "Malayalam",
        "Tamil": "Tamil",
        "Telugu": "Telugu",
        "Kannada": "Kannada",
        "Hindi": "Hindi",
    }

    def _construct_multilingual_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        language: str,
        answer_mode: str,
    ) -> tuple:
        """
        Construct an enhanced multilingual prompt for the LLM with better instructions.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            language: Document language code
            answer_mode: Answer format mode
            
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        language_name = get_language_name(language)
        
        mode_instructions = {
            "concise": "Provide a focused answer in no more than three sentences. Only include the most critical facts.",
            "detailed_with_citations": (
                "Provide a comprehensive answer. Include inline source citations in brackets referencing the source filename and chunk index, "
                "for example [source: filename.txt #3]. Mention every relevant detail from the context."
            ),
            "bullet_summary": "Respond as a bullet list. Each bullet should contain one key fact with supporting detail.",
            "step_by_step": "Respond as numbered steps that lead the reader from context to conclusion. Ensure the reasoning is explicit.",
        }
        mode_instruction = mode_instructions.get(
            answer_mode,
            mode_instructions["detailed_with_citations"],
        )

        # Enhanced system prompt with more specific instructions
        system_prompt = f"""You are an expert multilingual assistant specialized in South Indian languages and document analysis. 
Your primary task is to provide accurate, comprehensive, and contextually appropriate answers based on the provided document content.

CRITICAL INSTRUCTIONS:
1. LANGUAGE REQUIREMENT: You MUST respond EXCLUSIVELY in {language_name} ({language}). Do not mix languages.
2. CONTEXT USAGE: Use ALL available context chunks provided below. Each chunk may contain different but relevant information.
3. COMPREHENSIVENESS: Provide detailed, thorough answers. Include:
   - All relevant facts, figures, names, dates, and technical details from the context
   - Multiple perspectives if the context presents different viewpoints
   - Complete information even if it spans across multiple context chunks
4. ACCURACY: Base your answer ONLY on the provided context. Do not add information not present in the context.
5. STRUCTURE: Organize your answer logically:
   - Start with a direct answer to the question
   - Provide supporting details and evidence from the context
   - Include specific examples, numbers, or quotes when available
6. CULTURAL SENSITIVITY: Maintain appropriate cultural context and linguistic nuances for {language_name}
7. COMPLETENESS: If the question has multiple parts, address all parts comprehensively
8. CLARITY: Write clearly and concisely while being thorough
9. ANSWER MODE: {mode_instruction}

Language: {language_name} ({language})
Response Language: {language_name} ONLY"""

        # Format context with metadata - limit to top 3 chunks to reduce size
        # Filter duplicates and limit chunks
        unique_chunks = []
        seen_texts = set()
        
        # Limit to top 3 chunks for smaller payload
        limited_chunks = context_chunks[:3]
        
        for chunk in limited_chunks:
            chunk_text = chunk['text'].strip()
            
            # Truncate very long chunks
            if len(chunk_text) > 500:
                chunk_text = chunk_text[:500] + "..."
            
            # Normalize for comparison
            normalized = re.sub(r'\s+', ' ', chunk_text.lower())
            
            # Skip if exact duplicate
            if normalized in seen_texts:
                continue
            
            # Check for high similarity
            is_duplicate = False
            for seen_text in seen_texts:
                if len(normalized) > 50 and len(seen_text) > 50:
                    text_words = set(normalized.split())
                    seen_words = set(seen_text.split())
                    if len(text_words) > 0 and len(seen_words) > 0:
                        intersection = len(text_words & seen_words)
                        union = len(text_words | seen_words)
                        similarity = intersection / union if union > 0 else 0
                        if similarity > 0.85:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                unique_chunks.append({**chunk, 'text': chunk_text})
                seen_texts.add(normalized)
        
        # Use unique chunks only, format more compactly
        context_text = ""
        for i, chunk in enumerate(unique_chunks, 1):
            source_label = f"{chunk['filename']} #{chunk['chunk_index']}"
            context_text += f"Context {i} (source: {source_label}): {chunk['text']}\n\n"

        # Simplified, more compact prompt to reduce size
        user_prompt = f"""Context:
{context_text.strip()}

Question: {query}

Answer in {language_name} based on the context above. Follow the instructions for the selected answer mode: {mode_instruction}."""

        return system_prompt, user_prompt
    
    def _make_api_request(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        Make API request to Gemini with retry logic and better error handling.
        
        Args:
            prompt: Combined prompt (system + user)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Optional[str]: Generated response or None if error
        """
        # Validate API key format
        if not self.api_key or len(self.api_key) < 20:
            print(f"ERROR: Invalid API key format (length: {len(self.api_key) if self.api_key else 0})")
            return None
        
        # Log request details
        prompt_size = len(prompt)
        print(f"API Request - Model: {self.model}, Prompt: {prompt_size/1024:.1f}KB")
        
        # Warn if prompt is very large
        if prompt_size > 30000:  # ~30KB
            print(f"⚠️ Large prompt detected ({prompt_size/1024:.1f}KB), this may cause issues")
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                # Generate content using Gemini API
                response = self.client.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.1,
                        "max_output_tokens": 1000,
                        "top_p": 0.9,
                    }
                )
                
                if response and response.text:
                    return response.text.strip()
                else:
                    print(f"Unexpected response format: {response}")
                    return None
                    
            except Exception as e:
                error_msg = str(e)
                print(f"Gemini API error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                
                # Handle model not found errors (404)
                if "404" in error_msg or "not found" in error_msg.lower() or "not supported" in error_msg.lower():
                    # Try to get available models
                    try:
                        available_models = self.get_available_models()
                        model_error = f"Model '{self.model}' is not available. Available models: {', '.join(available_models[:5])}..."
                    except:
                        model_error = f"Model '{self.model}' is not available. Please check the model name."
                    return f"Model error: {model_error}"
                
                # Handle specific error types
                if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2
                        print(f"Rate limited. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Rate limit exceeded after {max_retries} attempts")
                        return None
                elif "401" in error_msg or "invalid" in error_msg.lower() or "unauthorized" in error_msg.lower() or "api key" in error_msg.lower():
                    print(f"Authentication error: Invalid API key")
                    return f"Authentication error: Invalid API key. Please check your API key in the sidebar settings."
                elif "400" in error_msg or "bad request" in error_msg.lower():
                    print(f"Bad request error: {error_msg}")
                    # Try with smaller prompt
                    if len(prompt) > 15000:
                        # Truncate prompt
                        prompt = prompt[:15000] + "\n\n[Prompt truncated]"
                        if attempt < max_retries - 1:
                            continue
                    return None
                else:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 1
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    return f"API error: {error_msg[:200]}"
        
        return None
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        language: str,
        answer_mode: str = "detailed_with_citations",
    ) -> str:
        """
        Generate an answer using Gemini API with enhanced error handling.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            language: Document language code
            answer_mode: Answer format mode
            
        Returns:
            str: Generated answer
        """
        try:
            if not context_chunks:
                return self._get_no_context_response(language)
            
            # Validate query
            if not query or not query.strip():
                return self._get_no_context_response(language)
            
            # Construct prompt
            system_prompt, user_prompt = self._construct_multilingual_prompt(
                query, context_chunks, language, answer_mode
            )
            
            # Combine system and user prompts for Gemini
            # Gemini uses a single prompt, so we combine system instructions with user content
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Limit total prompt to 15000 chars for reliability
            total_length = len(combined_prompt)
            max_total_length = 15000
            
            if total_length > max_total_length:
                print(f"⚠️ Prompt too long ({total_length} chars), truncating to {max_total_length}...")
                
                # Truncate context more aggressively
                if "Context:" in combined_prompt:
                    # Extract query first (most important)
                    if "Question:" in combined_prompt:
                        query_part = combined_prompt.split("Question:")[1]
                        # Calculate available space for context
                        available_for_context = max_total_length - len(system_prompt) - len(query_part) - 500
                        
                        if available_for_context > 1000:
                            # Extract and truncate context
                            context_part = combined_prompt.split("Context:")[1].split("Question:")[0]
                            if len(context_part) > available_for_context:
                                context_part = context_part[:available_for_context] + "\n\n[Context truncated]"
                            
                            combined_prompt = f"{system_prompt}\n\nContext:\n{context_part}\n\nQuestion:{query_part}"
                        else:
                            # Too little space, use minimal context
                            combined_prompt = f"{system_prompt}\n\nQuestion:{query_part}"
                
                print(f"✅ Optimized prompt size: {len(combined_prompt)} chars")
            
            # Make API request with retries
            response = self._make_api_request(combined_prompt, max_retries=3)
            
            if response and response.strip():
                # Check if response is an error message
                if response.startswith("Connection failed") or response.startswith("Authentication error") or response.startswith("API error") or response.startswith("Request error") or response.startswith("Unexpected error"):
                    # It's an error message, return it with language-specific wrapper
                    error_msg = response
                    language_error = self._get_error_response(language)
                    return f"{language_error}\n\nTechnical details: {error_msg}"
                
                # Clean up response
                cleaned_response = response.strip()
                return cleaned_response
            else:
                return self._get_error_response(language)
                
        except Exception as e:
            print(f"Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            return self._get_error_response(language)
    
    def translate_text(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
    ) -> str:
        """
        Translate the provided text into the requested language.
        
        Args:
            text: Text to translate
            target_language: Target language (can be code like 'en' or name like 'English')
            source_language: Optional source language name for context
        """
        if not text or not text.strip():
            return ""

        # Convert language code to name if needed
        if len(target_language) <= 3:
            # It's a language code, convert to name
            target_lang_name = get_language_name(target_language)
            if target_lang_name == 'Unknown':
                target_lang_name = target_language
        else:
            # It's already a language name
            target_lang_name = target_language
        
        # Check if target language is supported
        if target_lang_name not in self.TRANSLATABLE_LANGUAGES:
            return text

        try:
            prompt = (
                f"You are a professional translation engine. Your task is to translate text into {target_lang_name}. "
                f"IMPORTANT: You MUST translate the text. Do NOT return the original text. "
                f"Provide ONLY the translated text in {target_lang_name}, without any additional commentary, explanations, or notes. "
                "Preserve the original meaning, tone, and structure. Keep all numbers, dates, and proper nouns as they are.\n\n"
                f"Translate the following text to {target_lang_name}:\n\n{text}"
            )
            
            response = self._make_api_request(prompt, max_retries=3)
            translated = response.strip() if response else ""
            
            # If translation failed or returned empty, return empty string
            if not translated:
                print(f"Translation returned empty response for text: {text[:50]}...")
                return ""
            
            # Check if the translation is actually different from the original
            if translated.strip() == text.strip():
                print(f"Warning: Translation returned same text as original. This might indicate the text is already in {target_lang_name}.")
            
            return translated
        except Exception as exc:
            print(f"Translation failed: {exc}")
            return ""

    def _get_no_context_response(self, language: str) -> str:
        """Get response when no context is available."""
        responses = {
            'hi': "क्षमा कीजिए, इस प्रश्न का उत्तर देने के लिए दस्तावेज़ में पर्याप्त जानकारी नहीं मिली।",
            'ml': "ക്ഷമിക്കണം, ഈ ചോദ്യത്തിന് ഉത്തരം നൽകാൻ ആവശ്യമായ വിവരങ്ങൾ രേഖയിൽ കണ്ടെത്താൻ കഴിഞ്ഞില്ല.",
            'ta': "மன்னிக்கவும், இந்த கேள்விக்கு பதில் அளிக்க தேவையான தகவல்களை ஆவணத்தில் காண முடியவில்லை.",
            'te': "క్షమించండి, ఈ ప్రశ్నకు సమాధానం ఇవ్వడానికి అవసరమైన సమాచారం పత్రంలో కనుగొనబడలేదు.",
            'kn': "ಕ್ಷಮಿಸಿ, ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ನೀಡಲು ಅಗತ್ಯವಾದ ಮಾಹಿತಿಯನ್ನು ದಾಖಲೆಯಲ್ಲಿ ಕಂಡುಹಿಡಿಯಲಾಗಲಿಲ್ಲ.",
            'tcy': "ಕ್ಷಮಿಸಿ, ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ನೀಡಲು ಅಗತ್ಯವಾದ ಮಾಹಿತಿಯನ್ನು ದಾಖಲೆಯಲ್ಲಿ ಕಂಡುಹಿಡಿಯಲಾಗಲಿಲ್ಲ.",
            'en': "Sorry, I couldn't find sufficient information in the document to answer this question."
        }
        return responses.get(language, responses.get(language, responses['en']))
    
    def _get_error_response(self, language: str) -> str:
        """Get response when there's an error."""
        responses = {
            'hi': "क्षमा कीजिए, उत्तर बनाने में त्रुटि हुई। कृपया फिर से प्रयास करें।",
            'ml': "ക്ഷമിക്കണം, ഉത്തരം സൃഷ്ടിക്കുന്നതിൽ ഒരു പിശക് സംഭവിച്ചു. ദയവായി വീണ്ടും ശ്രമിക്കുക.",
            'ta': "மன்னிக்கவும், பதிலை உருவாக்குவதில் பிழை ஏற்பட்டது. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.",
            'te': "క్షమించండి, సమాధానాన్ని సృష్టించడంలో లోపం సంభవించింది. దయచేసి మళ్లీ ప్రయత్నించండి.",
            'kn': "ಕ್ಷಮಿಸಿ, ಉತ್ತರವನ್ನು ರಚಿಸುವಲ್ಲಿ ದೋಷ ಸಂಭವಿಸಿದೆ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
            'tcy': "ಕ್ಷಮಿಸಿ, ಉತ್ತರವನ್ನು ರಚಿಸುವಲ್ಲಿ ದೋಷ ಸಂಭವಿಸಿದೆ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
            'en': "Sorry, there was an error generating the response. Please try again."
        }
        return responses.get(language, responses.get(language, responses['en']))
    
    def test_connection(self) -> bool:
        """
        Test the connection to Gemini API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            test_prompt = "Hello, please respond with 'Connection successful'."
            response = self._make_api_request(test_prompt)
            return response is not None and "successful" in response.lower()
            
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get list of available Gemini models.
        Tries to fetch from API, falls back to hardcoded list.
        
        Returns:
            List[str]: List of available model names
        """
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key and genai:
                genai.configure(api_key=api_key)
                models = genai.list_models()
                available = [
                    m.name.split('/')[-1] 
                    for m in models 
                    if 'generateContent' in m.supported_generation_methods
                ]
                # Filter to main models (exclude previews and experimental unless specifically requested)
                # Put gemini-2.5-flash first as it's the default
                main_models = [
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-3-pro-preview",
                    "gemini-2.5-flash-lite",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                    "gemini-pro-latest",
                    "gemini-flash-latest",
                ]
                # Return intersection of available and main models
                result = [m for m in main_models if m in available]
                if result:
                    return result
        except Exception as e:
            print(f"Warning: Could not fetch models from API: {e}")
        
        # Fallback to hardcoded list (gemini-2.5-flash is default, so it's first)
        return [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-3-pro-preview",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-pro-latest",
            "gemini-flash-latest",
        ]
    
    def set_model(self, model: str) -> bool:
        """
        Set the Gemini model to use.
        
        Args:
            model: Model name
            
        Returns:
            bool: True if model is valid, False otherwise
        """
        available_models = self.get_available_models()
        if model in available_models:
            self.model = model
            self.client = genai.GenerativeModel(self.model)
            return True
        else:
            print(f"Invalid model: {model}. Available models: {available_models}")
            return False


# Alias for backward compatibility (if needed)
GroqHandler = GeminiHandler
