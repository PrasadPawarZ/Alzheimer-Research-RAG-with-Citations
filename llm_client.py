"""
Thin wrapper so the rest of the app doesn't care whether we're calling
Groq or Gemini. Both have usable free tiers as of writing.

Add a new provider by implementing `_call_<name>` and wiring it into `chat()`.
"""
import config


def chat(system_prompt: str, user_prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    provider = config.LLM_PROVIDER
    if provider == "groq":
        return _call_groq(system_prompt, user_prompt, temperature, max_tokens)
    elif provider == "gemini":
        return _call_gemini(system_prompt, user_prompt, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'groq' or 'gemini'.")


def _call_groq(system_prompt, user_prompt, temperature, max_tokens):
    from groq import Groq

    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys")

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


def _call_gemini(system_prompt, user_prompt, temperature, max_tokens):
    import google.generativeai as genai

    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey")

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
    return resp.text.strip()
