"""
Token calculator module for calculating tokens and cost for different models.
"""
import tiktoken
from transformers import AutoTokenizer
from logging_utils import log
from config import VERBOSE_DETAILED

class TokenCalculator:
    """
    Class for calculating tokens and costs across different model types.
    Handles OpenAI (tiktoken) and other models (HuggingFace tokenizers).
    """
    
    def __init__(self, requestCostPerToken, responseCostPerToken, model=None):
        """
        Initialize with costs per token and optional model name.
        
        Args:
            requestCostPerToken (float): Cost per token for requests
            responseCostPerToken (float): Cost per token for responses  
            model (str, optional): Model name used for tokenization
        """
        # Costs are stored as dollars per token (NOT per 1000 tokens)
        self.requestCostPerToken = requestCostPerToken / 1000000  # Convert from per million
        self.responseCostPerToken = responseCostPerToken / 1000000  # Convert from per million
        self.model = model
        self._tokenizers = {}  # Cache for tokenizers to avoid reloading

    def calculate_tokens(self, text: str, tokenModel=None) -> int:
        """
        Calculate the number of tokens in the given text.
        
        Args:
            text (str): The text to calculate tokens for
            tokenModel (str, optional): Model to use for tokenization, overrides instance model
            
        Returns:
            int: Number of tokens in the text
        """
        if not text:
            return 0
            
        # Choose the model to use for tokenization
        model = tokenModel or self.model
        
        # Model must be defined
        if model is None:
            raise ValueError("Model was not defined. Unable to calculate tokens.")

        try:
            # Check if the model is a tiktoken encoding name
            tiktoken_encodings = ["cl100k_base", "p50k_base", "r50k_base", "p50k_edit", "r50k_edit"]
            
            # OpenAI models use tiktoken
            if "gpt" in model.lower() or model in tiktoken_encodings:
                if model not in self._tokenizers:
                    try:
                        if model in tiktoken_encodings:
                            # Use the encoding name directly
                            self._tokenizers[model] = tiktoken.get_encoding(model)
                        else:
                            # Try to get model-specific encoding
                            self._tokenizers[model] = tiktoken.encoding_for_model(model)
                    except KeyError:
                        # Fall back to cl100k_base for unknown OpenAI models
                        log(f"Unknown model {model}, falling back to cl100k_base encoding", "warning", VERBOSE_DETAILED)
                        self._tokenizers[model] = tiktoken.get_encoding("cl100k_base")
                
                encoding = self._tokenizers[model]
                return len(encoding.encode(text))
            # Handle Grok models specially, use cl100k_base encoding
            elif "grok" in model.lower():
                if model not in self._tokenizers:
                    # Use cl100k_base encoding for Grok models (similar to GPT-4)
                    log(f"Using cl100k_base encoding for Grok model {model}", "info", VERBOSE_DETAILED)
                    self._tokenizers[model] = tiktoken.get_encoding("cl100k_base")
                
                encoding = self._tokenizers[model]
                return len(encoding.encode(text))
            else:
                # Other models use HuggingFace's AutoTokenizer
                if model not in self._tokenizers:
                    try:
                        self._tokenizers[model] = AutoTokenizer.from_pretrained(model)
                    except Exception as e:
                        log(f"Error loading tokenizer for {model}: {e}", "error")
                        # Use a fallback tokenizer
                        log("Using fallback tokenizer (gpt2)", "warning", VERBOSE_DETAILED)
                        self._tokenizers[model] = AutoTokenizer.from_pretrained("gpt2")
                
                tokenizer = self._tokenizers[model]
                # Some tokenizers don't have tokenize method
                if hasattr(tokenizer, 'tokenize'):
                    return len(tokenizer.tokenize(text))
                else:
                    # Use encode method instead
                    return len(tokenizer.encode(text))
                    
        except Exception as e:
            log(f"Error calculating tokens: {e}", "error")
            # Estimate tokens (rough approximation)
            return len(text.split()) * 4 // 3  # ~4/3 tokens per word

    def calculate_cost(self, tokenCount, isRequest=True) -> float:
        """
        Calculate the cost for a given number of tokens.
        
        Args:
            tokenCount (int): Number of tokens
            isRequest (bool): Whether this is for a request (True) or response (False)
            
        Returns:
            float: Cost in USD
        """
        costPerToken = self.requestCostPerToken if isRequest else self.responseCostPerToken
        return tokenCount * costPerToken

    def estimate_prompt_cost(self, prompt_text, model_name=None, is_chat=True):
        """
        Estimate the cost of sending a prompt and receiving a typical response.
        
        NOTE: This function is currently UNUSED in the main codebase but kept for
        potential future cost estimation features.
        
        Args:
            prompt_text (str): The prompt text
            model_name (str, optional): Model to use, defaults to instance model
            is_chat (bool): Whether this is a chat model (affects token counting)
            
        Returns:
            tuple: (request_tokens, request_cost, estimated_response_tokens, estimated_response_cost, total_estimated_cost)
        """
        # Count request tokens
        request_tokens = self.calculate_tokens(prompt_text, model_name)
        request_cost = self.calculate_cost(request_tokens, True)
        
        # Estimate response tokens (typically 1.5x the request for general responses)
        estimated_response_tokens = min(request_tokens * 1.5, 1000)  # Cap at 1000 tokens
        estimated_response_cost = self.calculate_cost(estimated_response_tokens, False)
        
        # Total estimated cost
        total_estimated_cost = request_cost + estimated_response_cost
        
        return (request_tokens, request_cost, estimated_response_tokens, estimated_response_cost, total_estimated_cost)
