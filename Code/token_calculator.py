import tiktoken
from transformers import AutoTokenizer
from logging_utils import log
from config import VERBOSE_DETAILED

class TokenCalculator:
    def __init__(self, requestCostPerToken, responseCostPerToken, model=None):
        self.requestCostPerToken = requestCostPerToken / 1000000
        self.responseCostPerToken = responseCostPerToken / 1000000
        self.model = model
        self._tokenizers = {}

    def calculate_tokens(self, text: str, tokenModel=None) -> int:
        if not text:
            return 0
            
        model = tokenModel or self.model
        
        if model is None:
            raise ValueError("Model was not defined. Unable to calculate tokens.")

        try:
            tiktoken_encodings = ["cl100k_base", "p50k_base", "r50k_base", "p50k_edit", "r50k_edit"]
            
            if "gpt" in model.lower() or model in tiktoken_encodings:
                if model not in self._tokenizers:
                    try:
                        if model in tiktoken_encodings:
                            self._tokenizers[model] = tiktoken.get_encoding(model)
                        else:
                            self._tokenizers[model] = tiktoken.encoding_for_model(model)
                    except KeyError:
                        log(f"Unknown model {model}, falling back to cl100k_base encoding", "warning", VERBOSE_DETAILED)
                        self._tokenizers[model] = tiktoken.get_encoding("cl100k_base")
                
                encoding = self._tokenizers[model]
                return len(encoding.encode(text))
            elif "grok" in model.lower():
                if model not in self._tokenizers:
                    log(f"Using cl100k_base encoding for Grok model {model}", "info", VERBOSE_DETAILED)
                    self._tokenizers[model] = tiktoken.get_encoding("cl100k_base")
                
                encoding = self._tokenizers[model]
                return len(encoding.encode(text))
            else:
                if model not in self._tokenizers:
                    try:
                        self._tokenizers[model] = AutoTokenizer.from_pretrained(model)
                    except Exception as e:
                        log(f"Error loading tokenizer for {model}: {e}", "error")
                        log("Using fallback tokenizer (gpt2)", "warning", VERBOSE_DETAILED)
                        self._tokenizers[model] = AutoTokenizer.from_pretrained("gpt2")
                
                tokenizer = self._tokenizers[model]
                if hasattr(tokenizer, 'tokenize'):
                    return len(tokenizer.tokenize(text))
                else:
                    return len(tokenizer.encode(text))
                    
        except Exception as e:
            log(f"Error calculating tokens: {e}", "error")
            return len(text.split()) * 4 // 3

    def calculate_cost(self, tokenCount, isRequest=True) -> float:
        costPerToken = self.requestCostPerToken if isRequest else self.responseCostPerToken
        return tokenCount * costPerToken

    def estimate_prompt_cost(self, prompt_text, model_name=None, is_chat=True):
        request_tokens = self.calculate_tokens(prompt_text, model_name)
        request_cost = self.calculate_cost(request_tokens, True)
        
        estimated_response_tokens = min(request_tokens * 1.5, 1000)
        estimated_response_cost = self.calculate_cost(estimated_response_tokens, False)
        
        total_estimated_cost = request_cost + estimated_response_cost
        
        return (request_tokens, request_cost, estimated_response_tokens, estimated_response_cost, total_estimated_cost)
