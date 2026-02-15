from token_calculator import TokenCalculator

class LLM:
    def __init__(self, model, temperature, requestCostPerToken, responseCostPerToken, tokenModel=None):
        self.model = model
        self.temperature = temperature
        self.history = []

        self.tokenCalculator = TokenCalculator(
            requestCostPerToken, responseCostPerToken, tokenModel or model
        )

    def append_to_history(self, role, message):
        self.history.append({"role": role, "content": message})
    
    def clear_history(self):
        self.history = [msg for msg in self.history if msg["role"] == "system"]
    
    def get_last_message(self, role=None):
        if not self.history:
            return None
            
        if role:
            for msg in reversed(self.history):
                if msg["role"] == role:
                    return msg
            return None
        else:
            return self.history[-1]
            
    def calculate_history_tokens(self):
        return sum(
            self.tokenCalculator.calculate_tokens(
                msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
            )
            for msg in self.history
        )