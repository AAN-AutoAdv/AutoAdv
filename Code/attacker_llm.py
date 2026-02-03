import random
import traceback  # Import traceback for logging

# Make sure OpenAI is imported if used
try:
    from openai import OpenAI
except ImportError:
    # Handle case where openai library might not be installed if only using other APIs
    OpenAI = None
    pass
# Import necessary items from local modules
from llm_base import LLM
from logging_utils import (
    log,
    VERBOSE_DETAILED,
    VERBOSE_NORMAL,
)  # Ensure all levels used are imported
from utils import (
    check_api_key_existence,
    api_call_with_retry,
)  # categorize_prompt, get_specialized_strategy removed as they should be centralized
from config import ATTACKER_MODELS  # Import ATTACKER_MODELS
from grok_client import GrokClient  # Import your Grok client implementation
from temperature_manager import TemperatureManager  # Import TemperatureManager


class AttackerLLM(LLM):
    def __init__(
        self,
        temperature=0.7,
        instructions=None,
        followup_instructions=None,
        attacker_model_key="gpt4o-mini",
    ):
        self.temperature = temperature
        self.initial_instructions = instructions
        self.followup_instructions = followup_instructions
        self.model_key = attacker_model_key
        self.history = []
        self.using_followup = False
        
        from config import DEFAULT_CONFIG
        self.temp_manager = TemperatureManager(
            initial_temperature=temperature,
            min_temp=DEFAULT_CONFIG.get("temperature_min", 0.1),
            max_temp=DEFAULT_CONFIG.get("temperature_max", 1.5),
            success_threshold=DEFAULT_CONFIG.get("strongreject_threshold", 0.6)
        )

        if "grok" in self.model_key.lower():
            self.client = GrokClient()
            self.model_config = ATTACKER_MODELS.get(
                self.model_key,
                {
                    "name": "Grok Model",
                    "request_cost": 0.00001,
                    "response_cost": 0.00001,
                    "token_limit": 4096,
                    "provider": "xai",
                },
            )
        else:
            if attacker_model_key not in ATTACKER_MODELS:
                raise ValueError(
                    f"Unknown attacker model: {attacker_model_key}. Available options: {', '.join(ATTACKER_MODELS.keys())}"
                )

            model_config = ATTACKER_MODELS[attacker_model_key]
            self.api_type = model_config.get(
                "api", "openai"
            )  # Default to openai if not specified

            # set the initial parameters and variables
            super().__init__(
                model=model_config["name"],
                temperature=temperature,
                requestCostPerToken=model_config["request_cost"],
                responseCostPerToken=model_config["response_cost"],
                tokenModel=model_config.get("token_model"),  # Use get for safety
            )

            self.client = self._initialize_api_client()  # Use helper method

        self.system_prompt = instructions
        self.followup_prompt = followup_instructions
        if self.system_prompt:
            self.append_to_history("system", self.system_prompt)
        else:
            log("Attacker: missing system prompt", "warning")


    def _initialize_api_client(self):
        log(
            f"Initializing attacker client for API type: {self.api_type}",
            "debug",
            VERBOSE_DETAILED,
        )
        if self.api_type == "openai":
            if not OpenAI:
                raise ImportError("OpenAI library not installed.")
            return OpenAI(api_key=check_api_key_existence("OPENAI_API_KEY"))
        elif self.api_type == "together":
            # Together AI also uses the OpenAI client format
            if not OpenAI:
                raise ImportError("OpenAI library not installed (needed for Together).")
            return OpenAI(
                api_key=check_api_key_existence("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1",
            )
        else:
            raise ValueError(f"Unsupported API type for attacker: {self.api_type}")

    def _log_exception(self, context, exc):
                log(
                    f"Attacker: {context} exception: {type(exc).__name__}: {exc}",
                    "error",
                )
        log(traceback.format_exc(), "error", VERBOSE_DETAILED)

    def _extract_chat_content(self, response, context):
        if not response:
            log(f"Attacker: {context} returned empty response", "error")
            return None

        if not hasattr(response, "choices") or not response.choices:
            log(f"Attacker: {context} response missing choices", "error")
            return None

        try:
            message = response.choices[0].message
            content = getattr(message, "content", None)
            if not content or not content.strip():
                log(f"Attacker: {context} returned empty content", "warning", VERBOSE_DETAILED)
                return None
            return content.strip()
        except (AttributeError, TypeError, IndexError) as e:
            log(f"Attacker: {context} response access error: {e}", "error")
            return None

    def rewrite(self, prompt):
        if "grok" in self.model_key.lower():
            # Make sure we're using the correct system prompt variable name
            system_prompt = (
                self.initial_instructions
            )  # Make sure this matches your variable name

            # Debug logging to verify the system prompt is being passed
            log(
                f"Using system prompt for Grok rewrite: {system_prompt[:100]}...",
                "debug",
                VERBOSE_DETAILED,
            )

            # Pass the system prompt to the _generate_with_grok method
            (
                rewritten_prompt,
                request_tokens,
                response_tokens,
                request_cost,
                response_cost,
            ) = self._generate_with_grok(system_prompt, prompt)
            print(
                f"Rewritten prompt: {rewritten_prompt}"
            )  # Debug print to verify the rewritten prompt
            print(f"Prompt: {prompt}")
            print(f"System Prompt: {system_prompt}")
            return (
                rewritten_prompt,
                request_tokens,
                response_tokens,
                request_cost,
                response_cost,
            )
        else:
            original_prompt = prompt

            # Prepare history for rewrite: only system prompt + current user prompt
            rewrite_history = [msg for msg in self.history if msg["role"] == "system"]
            rewrite_history.append({"role": "user", "content": original_prompt})

            log(
                f"Attacker ({self.model}) rewriting prompt with temp {self.temperature}",
                "debug",
                VERBOSE_DETAILED,
            )
            log(
                f"History being sent for rewrite: {rewrite_history}",
                "debug",
                VERBOSE_DETAILED + 1,
            )  # Log history

            response = None  # Initialize response variable

            try:
                # Prepare arguments for API call
                api_args = {
                    "model": self.model,
                    "messages": rewrite_history,  # Send the specifically prepared history
                    "max_tokens": 250,
                    "n": 1,
                    "temperature": self.temperature,
                }

                # Generate a response from the attacker model
                response = api_call_with_retry(
                    self.client.chat.completions.create,  # Pass the method instead of the client
                    model=self.model,
                    messages=rewrite_history,
                    max_tokens=250,
                    n=1,
                    temperature=self.temperature,
                )
                # --- Start: Response Validation ---
                if not response:
                    log(f"API call returned None unexpectedly after retries.", "error")
                    return None, 0, 0, 0.0, 0.0

                if not hasattr(response, "choices") or not response.choices:
                    log(
                        f"API call succeeded but response invalid (no choices attribute or empty choices list): {response}",
                        "error",
                    )
                    return None, 0, 0, 0.0, 0.0

                # Check message and content exist
                try:
                    message = response.choices[0].message
                    if (
                        not message
                        or not hasattr(message, "content")
                        or message.content is None
                    ):
                        log(
                            f"API call succeeded but response structure invalid (no message or content): {response}",
                            "error",
                        )
                        return None, 0, 0, 0.0, 0.0

                    # Check for empty content in response
                    if not message.content.strip():
                        log(
                            "API returned empty content response.",
                            "warning",
                            VERBOSE_DETAILED,
                        )
                        return None, 0, 0, 0.0, 0.0
                    else:
                        rewritten_prompt = message.content.strip()

                    if not rewritten_prompt:
                        log(
                            f"API call succeeded but response content is empty: {response}",
                            "warning",
                        )
                        # Treat empty rewrite as failure
                        return None, 0, 0, 0.0, 0.0

                except (AttributeError, TypeError, IndexError) as e:
                    log(
                        f"Error accessing response content: {e}. Response structure: {response}",
                        "error",
                    )
                    return None, 0, 0, 0.0, 0.0
                # --- End: Response Validation ---

                # Clean up potentially problematic self-references
                rewritten_prompt = rewritten_prompt.replace(
                    "As an AI assistant,", ""
                ).strip()
                rewritten_prompt = rewritten_prompt.replace(
                    "Okay, here is the rewritten prompt:", ""
                ).strip()
                rewritten_prompt = rewritten_prompt.replace(
                    "Okay, here's the rewritten prompt:", ""
                ).strip()


                # NOTE: We DO NOT modify self.history here. Rewrite is a one-off call.

                # Calculate tokens and costs based *only* on the prompt sent and response received
                # Request tokens = system prompt + user prompt that was sent
                request_text_parts = [
                    msg["content"] for msg in rewrite_history
                ]  # Use the history sent
                request_text_for_calc = "\n".join(request_text_parts)

                request_tokens = self.tokenCalculator.calculate_tokens(
                    request_text_for_calc
                )
                response_tokens = self.tokenCalculator.calculate_tokens(
                    rewritten_prompt
                )

                # Use usage info from response if available (more accurate)
                if hasattr(response, "usage") and response.usage:
                    prompt_tokens_api = getattr(response.usage, "prompt_tokens", None)
                    completion_tokens_api = getattr(
                        response.usage, "completion_tokens", None
                    )
                    if prompt_tokens_api is not None:
                        request_tokens = prompt_tokens_api
                    if completion_tokens_api is not None:
                        response_tokens = completion_tokens_api
                    pass

                request_cost = self.tokenCalculator.calculate_cost(
                    request_tokens, isRequest=True
                )
                response_cost = self.tokenCalculator.calculate_cost(
                    response_tokens, isRequest=False
                )

                return (
                    rewritten_prompt,
                    request_tokens,
                    response_tokens,
                    request_cost,
                    response_cost,
                )

            except Exception as e:
                self._log_exception("rewrite", e)
                return None, 0, 0, 0.0, 0.0

        return (
            rewritten_prompt,
            request_tokens,
            response_tokens,
            request_cost,
            response_cost,
        )

    def converse(self):
        if "grok" in self.model_key.lower():
            system_prompt = (
                self.followup_instructions
                if self.using_followup
                else self.initial_instructions
            )

            # Pass the system prompt to the _generate_with_grok method
            response, request_tokens, response_tokens, request_cost, response_cost = (
                self._generate_with_grok(system_prompt, self.history)
            )
            return (
                response,
                request_tokens,
                response_tokens,
                request_cost,
                response_cost,
            )
        else:
            # Ensure history has at least a system prompt and a user message
            if len(self.history) < 2 or self.history[-1]["role"] != "user":
                log("Attacker: invalid history state", "warning")
                # Optionally try to recover or just fail
                return None, 0, 0, 0.0, 0.0

            try:
                # Prepare arguments for API call
                response = api_call_with_retry(
                    self.client.chat.completions.create,  # Pass the method instead of the client
                    model=self.model,
                    messages=self.history,
                    max_tokens=250,
                    n=1,
                    temperature=self.temperature,
                )

                reply = self._extract_chat_content(response, "converse")
                if reply is None:
                    return None, 0, 0, 0.0, 0.0

                # Check if the response is a refusal
                refusal_phrases = [
                    "i'm sorry, but i cannot",
                    "i cannot assist",
                    "i can't help",
                    "i am unable to assist",
                    "i apologize, but i cannot",
                    "my purpose is to be helpful and harmless",
                    "violates my safety policies",
                ]

                reply_lower = reply.lower()
                if any(phrase in reply_lower for phrase in refusal_phrases):
                    log("Attacker: refusal in response", "warning", VERBOSE_DETAILED)
                    # Make response more concise to avoid token waste and signal failure clearly
                    reply = "I apologize, but I cannot continue with this line of conversation as it seems counter-productive."

                # Calculate tokens for the *entire conversation history* that was sent to the API
                request_text_for_calc = "\n".join(
                    [msg["content"] for msg in self.history]
                )
                request_tokens = self.tokenCalculator.calculate_tokens(
                    request_text_for_calc
                )
                response_tokens = self.tokenCalculator.calculate_tokens(reply)

                # Use usage info from response if available
                if hasattr(response, "usage") and response.usage:
                    prompt_tokens_api = getattr(response.usage, "prompt_tokens", None)
                    completion_tokens_api = getattr(
                        response.usage, "completion_tokens", None
                    )
                    if prompt_tokens_api is not None:
                        request_tokens = prompt_tokens_api
                    if completion_tokens_api is not None:
                        response_tokens = completion_tokens_api

                # Calculate token cost
                request_cost = self.tokenCalculator.calculate_cost(
                    request_tokens, isRequest=True
                )
                response_cost = self.tokenCalculator.calculate_cost(
                    response_tokens, isRequest=False
                )

                # Add the attacker's reply to internal history *after* calculating request tokens
                self.append_to_history("assistant", reply)

                return (
                    reply,
                    request_tokens,
                    response_tokens,
                    request_cost,
                    response_cost,
                )

            except Exception as e:
                self._log_exception("converse", e)
                return None, 0, 0, 0.0, 0.0  # failure values

        return response, request_tokens, response_tokens, request_cost, response_cost

    def _generate_with_grok(self, instructions, prompt_or_history):
        try:
            # Process the prompt or history to fit Grok's expected format
            if isinstance(prompt_or_history, list):
                # For conversation, format the history appropriately for Grok
                formatted_messages = self._format_history_for_grok(prompt_or_history)
            else:
                # For single prompt rewriting
                formatted_messages = [{"role": "user", "content": prompt_or_history}]

            # Call Grok API using the new client
            if instructions:
                full_messages = [{"role": "system", "content": instructions}] + formatted_messages
            else:
                full_messages = formatted_messages

            completion = self.client.generate(
                system_message=instructions,
                messages=formatted_messages,
                temperature=self.temperature,
            )

            # Extract response text and metadata
            response_text = completion.get("response", "")
            request_tokens = completion.get("usage", {}).get("prompt_tokens", 0)
            response_tokens = completion.get("usage", {}).get("completion_tokens", 0)


            # Calculate costs using the model_config that was set during initialization
            request_cost = request_tokens * self.model_config.get("request_cost", 0.00001)
            response_cost = response_tokens * self.model_config.get("response_cost", 0.00001)

            return (
                response_text,
                request_tokens,
                response_tokens,
                request_cost,
                response_cost,
            )

        except Exception as e:
            self._log_exception("grok_generate", e)
            return "", 0, 0, 0, 0

    def _format_history_for_grok(self, history):
        formatted_messages = []
        for message in history:
            formatted_messages.append(
                {"role": message["role"], "content": message["content"]}
            )
        return formatted_messages

    def set_followup_prompt(self):
        if not self.followup_prompt:
            log(
                "No followup prompt available, continuing with original system prompt",
                "warning",
                VERBOSE_DETAILED,
            )
            return False

        # Find and replace the system message in history
        system_message_updated = False
        for i, message in enumerate(self.history):
            if message["role"] == "system":
                if message["content"] != self.followup_prompt:
                    log(
                        "Switching to followup system prompt for later turns",
                        "debug",
                        VERBOSE_DETAILED,
                    )
                    self.history[i]["content"] = self.followup_prompt
                    self.system_prompt = self.followup_prompt
                    system_message_updated = True
                else:
                    # Already using follow-up prompt
                    log(
                        "Already using followup system prompt.",
                        "debug",
                        VERBOSE_DETAILED + 1,
                    )
                    system_message_updated = True
                break  # Assume only one system message at the start

        # If no system message found (shouldn't happen with current init), add one at the beginning
        if not system_message_updated:
            log(
                "No system message found in history, inserting followup prompt at the beginning.",
                "warning",
                VERBOSE_DETAILED,
            )
            self.history.insert(0, {"role": "system", "content": self.followup_prompt})
            self.system_prompt = self.followup_prompt
            system_message_updated = True

        log(
            f"Current system prompt after set_followup_prompt: {self.system_prompt[:100]}...",
            "debug",
            VERBOSE_DETAILED + 1,
        )
        return system_message_updated
    
    def adjust_temperature_smart(self, target_response, strategy="adaptive", original_prompt=None):
        # Use StrongREJECT for accurate scoring if enabled and original prompt is available
        from config import DEFAULT_CONFIG
        use_strongreject = DEFAULT_CONFIG.get("use_strongreject_for_temperature", True)
        
        if use_strongreject and original_prompt:
            try:
                from conversation import evaluate_with_strongreject
                success_score, eval_details = evaluate_with_strongreject(original_prompt, target_response)
            except Exception as e:
                log("StrongREJECT evaluation failed, using default score", "warning", VERBOSE_DETAILED)
                success_score = 0.5  # Default to neutral score
        else:
            success_score = 0.5  # Default to neutral score
        
        # Use TemperatureManager to adjust temperature
        new_temp = self.temp_manager.adjust_temperature(success_score, strategy)
        
        # Update our temperature
        self.temperature = new_temp
        
        return new_temp
    
    def get_temperature_history(self):
        return self.temp_manager.temperature_history
    
    def get_success_history(self):
        return self.temp_manager.success_history
