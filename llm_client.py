"""LLM provider wrapper with safe offline/extractive fallback support."""
from dataclasses import dataclass
from typing import Optional

import config


@dataclass(frozen=True)
class ProviderStatus:
    requested: str
    active: str
    has_key: bool
    message: str


def _is_placeholder(value: str) -> bool:
    lowered = (value or "").strip().lower()
    return not lowered or lowered.startswith("your_") or "key_here" in lowered


def resolve_provider() -> str:
    provider = config.LLM_PROVIDER
    if provider == "auto":
        if not _is_placeholder(config.GEMINI_API_KEY):
            return "gemini"
        if not _is_placeholder(config.GROQ_API_KEY):
            return "groq"
        return "extractive"
    if provider == "gemini":
        return "gemini" if not _is_placeholder(config.GEMINI_API_KEY) else "extractive"
    if provider == "groq":
        return "groq" if not _is_placeholder(config.GROQ_API_KEY) else "extractive"
    return provider


def has_generation_provider() -> bool:
    return resolve_provider() in {"gemini", "groq"}


def provider_status() -> ProviderStatus:
    active = resolve_provider()
    if active == "gemini":
        return ProviderStatus(config.LLM_PROVIDER, active, True, f"Gemini enabled ({config.GEMINI_MODEL}).")
    if active == "groq":
        return ProviderStatus(config.LLM_PROVIDER, active, True, f"Groq enabled ({config.GROQ_MODEL}).")
    return ProviderStatus(
        config.LLM_PROVIDER,
        "extractive",
        False,
        "No LLM key configured. The app will use extractive grounded snippets.",
    )


def chat(system_prompt: str, user_prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    provider = resolve_provider()
    if provider == "gemini":
        return _call_gemini(system_prompt, user_prompt, temperature, max_tokens)
    if provider == "groq":
        return _call_groq(system_prompt, user_prompt, temperature, max_tokens)
    if provider == "extractive":
        raise RuntimeError("No LLM provider is configured. Use extractive mode instead.")
    raise ValueError("Unknown LLM_PROVIDER. Use auto, gemini, groq, or extractive.")


def _call_gemini(system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    import google.generativeai as genai

    if _is_placeholder(config.GEMINI_API_KEY):
        raise RuntimeError("GEMINI_API_KEY is not set.")

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=config.GEMINI_MODEL,
        system_instruction=system_prompt,
    )
    resp = model.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return (resp.text or "").strip()


def _call_groq(system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    from groq import Groq

    if _is_placeholder(config.GROQ_API_KEY):
        raise RuntimeError("GROQ_API_KEY is not set.")

    client = Groq(api_key=config.GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()
