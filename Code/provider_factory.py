import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from config import PROVIDER_SPECS

PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "together": "Together",
    "xai": "xAI",
    "anthropic": "Anthropic",
}

SDK_DISPLAY_NAMES = {
    "openai_python": "openai-python",
    "anthropic_python": "anthropic-python",
}


def _normalize_provider_name(provider_name):
    if provider_name == "grok":
        return "xai"
    return provider_name


def resolve_provider_config(model_config, model_key=None):
    provider = _normalize_provider_name(
        model_config.get("provider") or model_config.get("api")
    )
    if provider not in PROVIDER_SPECS:
        raise ValueError(
            f"Unknown provider '{provider}' for model '{model_key or model_config.get('name', 'unknown')}'"
        )

    resolved_config = dict(PROVIDER_SPECS[provider])
    resolved_config.update(model_config)
    resolved_config["provider"] = provider
    return resolved_config


def get_provider_info(model_config, model_key=None):
    resolved_config = resolve_provider_config(model_config, model_key=model_key)
    sdk_family = resolved_config["sdk_family"]
    base_url = resolved_config.get("base_url")

    return {
        "model_key": model_key or resolved_config.get("name", "unknown"),
        "model_name": resolved_config.get("name", "Unknown"),
        "provider": resolved_config["provider"],
        "provider_display_name": PROVIDER_DISPLAY_NAMES.get(
            resolved_config["provider"], resolved_config["provider"]
        ),
        "sdk_family": sdk_family,
        "sdk_display_name": SDK_DISPLAY_NAMES.get(sdk_family, sdk_family),
        "base_url": base_url,
        "endpoint": base_url or "SDK default",
        "api_key_env": resolved_config["api_key_env"],
        "compat_mode": resolved_config["compat_mode"],
    }


def format_provider_summary(provider_info):
    return (
        f"Provider: {provider_info['provider_display_name']} | "
        f"SDK: {provider_info['sdk_display_name']} | "
        f"Endpoint: {provider_info['endpoint']} | "
        f"Key: {provider_info['api_key_env']}"
    )


def build_provider_client(model_config, model_key=None, api_key_resolver=None):
    resolved_config = resolve_provider_config(model_config, model_key=model_key)
    provider_info = get_provider_info(resolved_config, model_key=model_key)

    api_key_env = provider_info["api_key_env"]
    if api_key_resolver is None:
        api_key = os.getenv(api_key_env)
    else:
        api_key = api_key_resolver(api_key_env)

    provider = provider_info["provider"]
    if provider in {"openai", "together", "xai"}:
        if not OpenAI:
            raise ImportError(
                f"OpenAI library not installed (needed for provider '{provider_info['provider_display_name']}')."
            )

        client_kwargs = {"api_key": api_key}
        if provider != "openai" and provider_info["base_url"]:
            client_kwargs["base_url"] = provider_info["base_url"]
        client = OpenAI(**client_kwargs)
    elif provider == "anthropic":
        if not Anthropic:
            raise ImportError(
                "Anthropic library not installed. pip install anthropic"
            )
        client = Anthropic(api_key=api_key)
    else:
        raise ValueError(
            f"Unsupported provider '{provider}' for model '{provider_info['model_key']}'"
        )

    return client, provider_info
