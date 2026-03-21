from config import ATTACKER_MODELS, VERBOSE_DETAILED
from logging_utils import log
from provider_factory import (
    build_provider_client,
    format_provider_summary,
    get_provider_info,
)
from utils import check_api_key_existence


class XAIChatClient:
    def __init__(self, model_config=None, model_key="grok-3-mini-beta"):
        self.model_key = model_key
        self.model_config = model_config or ATTACKER_MODELS.get(model_key)
        if self.model_config is None:
            raise ValueError(f"Unknown xAI model key: {model_key}")

        provider_info = get_provider_info(self.model_config, model_key=self.model_key)
        if provider_info["provider"] != "xai":
            raise ValueError(
                f"Model '{self.model_key}' is not configured for xAI. Provider is '{provider_info['provider']}'."
            )

        self.client, self.provider_info = build_provider_client(
            self.model_config,
            model_key=self.model_key,
            api_key_resolver=check_api_key_existence,
        )

        log(
            f"xAI client initialized for {self.model_key}. {format_provider_summary(self.provider_info)}",
            "debug",
            VERBOSE_DETAILED,
        )

    def generate(
        self,
        system_message=None,
        messages=None,
        temperature=0.7,
        model="grok-3-mini-beta",
    ):
        if system_message:
            log(
                f"xAI system message: {system_message[:100]}...",
                "debug",
                VERBOSE_DETAILED,
            )
        else:
            log(
                "No system message provided for xAI API call",
                "warning",
                VERBOSE_DETAILED,
            )

        try:
            if system_message:
                full_messages = [{"role": "system", "content": system_message}] + (
                    messages or []
                )
            else:
                full_messages = messages or []

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
            log(f"Error calling xAI API: {e}", "error")
            return {
                "response": "",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
