import os
import sys
from openai import OpenAI

# Add proper import path for config
sys.path.insert(
    1,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Helpers"
    ),
)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VERBOSE_DETAILED
from logging_utils import log


class GrokClient:
    """
    Client for interacting with the Grok API.
    """

    def __init__(self):
        """Initialize the Grok client."""
        self.api_key = os.environ.get("XAI_API_KEY")
        if not self.api_key:
            log("XAI_API_KEY environment variable not set", "warning")

        self.client = OpenAI(
            base_url="https://api.x.ai/v1",
            api_key=self.api_key,
        )

    def generate(self, system_message=None, messages=None, temperature=0.7, model="grok-3-mini-beta"):
        """
        Generate a response using the Grok API.

        Args:
            system_message (str): System instructions
            messages (list): List of message dictionaries
            temperature (float): Sampling temperature
            model (str): Model name to use

        Returns:
            dict: Response from the Grok API
        """
        if system_message:
            log(
                f"System message being sent to Grok API: {system_message[:100]}...",
                "debug",
                VERBOSE_DETAILED,
            )
        else:
            log(
                "No system message provided for Grok API call",
                "warning",
                VERBOSE_DETAILED,
            )

        if not self.api_key:
            log("XAI_API_KEY not set. Cannot generate response.", "error")
            return {
                "response": "",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

        try:
            # Prepend system message to the messages list
            if system_message:
                full_messages = [{"role": "system", "content": system_message}] + (
                    messages or []
                )
            else:
                full_messages = messages or []

            # Create the chat completion
            completion = self.client.chat.completions.create(
                model=model,
                messages=full_messages,
                temperature=temperature,
            )

            return {
                "response": completion.choices[0].message.content,
                "reasoning": getattr(
                    completion.choices[0].message, "reasoning_content", None
                ),
                "usage": {
                    "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(
                        completion.usage, "completion_tokens", 0
                    ),
                    "reasoning_tokens": getattr(
                        getattr(completion.usage, "completion_tokens_details", {}),
                        "reasoning_tokens",
                        0,
                    ),
                },
            }

        except Exception as e:
            log(f"Error calling Grok API: {e}", "error")
            return {
                "response": "",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }


# Example usage of GrokClient
if __name__ == "__main__":
    grok_client = GrokClient()
    system_prompt = "You are a highly intelligent AI assistant."
    user_message = {"role": "user", "content": "What is 101*3?"}

    response = grok_client.generate(
        system_message=system_prompt, messages=[user_message], temperature=0.7
    )

    print("Reasoning Content:")
    print(response.get("reasoning", "<No reasoning content>"))

    print("\nFinal Response:")
    print(response["response"])

    print("\nNumber of completion tokens (input):")
    print(response["usage"].get("completion_tokens", 0))

    print("\nNumber of reasoning tokens (input):")
    print(response["usage"].get("reasoning_tokens", 0))
