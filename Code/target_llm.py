import traceback

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    pass
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None
    pass

from llm_base import LLM
from utils import check_api_key_existence, api_call_with_retry
from logging_utils import log
from config import TARGET_MODELS, VERBOSE_DETAILED


class TargetLLM(LLM):
    def __init__(
        self,
        temperature,
        target_model_key="llama3-8b",
        memory_enabled=False,
        model_config=None,
    ):
        if model_config:
            super().__init__(
                model=model_config["name"],
                temperature=temperature,
                requestCostPerToken=model_config["request_cost"],
                responseCostPerToken=model_config["response_cost"],
                tokenModel=model_config.get("token_model"),
            )
            self.model_key = target_model_key
            self.api_type = model_config.get(
                "api", "together"
            )
        else:
            if target_model_key not in TARGET_MODELS:
                raise ValueError(
                    f"Unknown target model: {target_model_key}. Available options: {', '.join(TARGET_MODELS.keys())}"
                )

            model_config = TARGET_MODELS[target_model_key]
            self.model_key = target_model_key
            self.api_type = model_config.get("api", "together")

            super().__init__(
                model=model_config["name"],
                temperature=temperature,
                requestCostPerToken=model_config["request_cost"],
                responseCostPerToken=model_config["response_cost"],
                tokenModel=model_config.get("token_model"),
            )

        self.memory_enabled = memory_enabled
        self.client = self._initialize_api_client()
        self.history = []

    def _initialize_api_client(self):
        log(f"Target: init client ({self.api_type})", "debug", VERBOSE_DETAILED)
        if self.api_type == "together":
            if not OpenAI:
                raise ImportError("OpenAI library not installed (needed for Together).")
            return OpenAI(
                api_key=check_api_key_existence("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1",
            )
        elif self.api_type == "openai":
            if not OpenAI:
                raise ImportError("OpenAI library not installed.")
            return OpenAI(api_key=check_api_key_existence("OPENAI_API_KEY"))
        elif self.api_type == "anthropic":
            if not Anthropic:
                raise ImportError(
                    "Anthropic library not installed. pip install anthropic"
                )
            return Anthropic(api_key=check_api_key_existence("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unsupported API type for target: {self.api_type}")

    def _log_exception(self, context, exc):
        log(
            f"Target: {context} exception ({self.model}): {type(exc).__name__}: {exc}",
            "error",
        )
        log(traceback.format_exc(), "error", VERBOSE_DETAILED)

    def converse(self, request):
        if not request:
            log("Target: empty request", "warning")
            return None, 0, 0, 0.0, 0.0

        try:
            messages_to_send = []
            if self.memory_enabled:
                if not self.history or self.history[-1].get("role") == "assistant":
                    self.append_to_history("user", request)
                elif self.history[-1].get("role") == "user":
                    self.history[-1]["content"] = request
                else:
                    self.append_to_history("user", request)

                messages_to_send = self.history
            else:
                messages_to_send = [{"role": "user", "content": request}]

            api_args = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": 1024,
            }
            response_content = None
            completion_details = None

            if self.api_type in ["together", "openai"]:
                api_args["messages"] = messages_to_send

                completion = api_call_with_retry(
                    self.client.chat.completions.create, **api_args
                )

                if completion.choices:
                    response_content = completion.choices[0].message.content

                completion_details = completion

            elif self.api_type == "anthropic":
                anthropic_messages = []
                system_prompt_content = None
                for msg in messages_to_send:
                    if msg["role"] == "system":
                        if "system" not in api_args:
                            api_args["system"] = msg["content"]
                        else:
                            api_args["system"] += "\n" + msg["content"]
                        continue

                    role = "user" if msg["role"] == "user" else "assistant"
                    anthropic_messages.append({"role": role, "content": msg["content"]})

                api_args["messages"] = anthropic_messages

                completion = self.client.messages.create(
                    **api_args
                )
                if completion.content:
                    response_content = completion.content[0].text
                completion_details = completion

            if response_content is None:
                log(f"Target: empty response content ({self.model})", "error")
                response_content = "[Model returned no content]"

            if not isinstance(response_content, str) or response_content.strip() == "":
                log(f"Target: empty or invalid response ({self.model})", "warning")
                response_content = "[Model returned empty or invalid response]"

            response_content = response_content.strip()

            if self.memory_enabled:
                if self.history and self.history[-1].get("role") == "user":
                    self.append_to_history("assistant", response_content)
                else:
                    log("Target: history state prevents assistant append", "warning", VERBOSE_DETAILED)

            request_text_for_calc = ""
            if self.api_type == "anthropic" and "system" in api_args:
                request_text_for_calc = (
                    api_args["system"]
                    + "\n"
                    + "\n".join([msg["content"] for msg in api_args["messages"]])
                )
            elif "messages" in api_args:
                request_text_for_calc = "\n".join(
                    [msg["content"] for msg in api_args["messages"]]
                )

            requestTokens = self.tokenCalculator.calculate_tokens(request_text_for_calc)
            responseTokens = self.tokenCalculator.calculate_tokens(response_content)

            if (
                completion_details
                and hasattr(completion_details, "usage")
                and completion_details.usage
            ):
                usage_info = completion_details.usage
                prompt_tokens_api = getattr(
                    usage_info,
                    "input_tokens",
                    getattr(usage_info, "prompt_tokens", None),
                )
                completion_tokens_api = getattr(
                    usage_info,
                    "output_tokens",
                    getattr(usage_info, "completion_tokens", None),
                )

                if prompt_tokens_api is not None:
                    requestTokens = prompt_tokens_api
                if completion_tokens_api is not None:
                    responseTokens = completion_tokens_api
                pass

            requestCost = self.tokenCalculator.calculate_cost(
                requestTokens, isRequest=True
            )
            responseCost = self.tokenCalculator.calculate_cost(
                responseTokens, isRequest=False
            )

            log(f"Target: response ok ({self.model})", "debug", VERBOSE_DETAILED)

            return (
                response_content,
                requestTokens,
                responseTokens,
                requestCost,
                responseCost,
            )

        except Exception as e:
            self._log_exception("converse", e)
            return None, 0, 0, 0.0, 0.0
