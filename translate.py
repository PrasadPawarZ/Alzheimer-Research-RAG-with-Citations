"""Language detection and optional translation helpers."""
from langdetect import DetectorFactory, LangDetectException, detect

import llm_client

DetectorFactory.seed = 0

LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
}


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def language_name(code: str) -> str:
    return LANG_NAMES.get(code, code or "unknown")


def translate(text: str, target_lang_code: str) -> str:
    if target_lang_code == "en" or not text.strip() or not llm_client.has_generation_provider():
        return text

    lang_name = language_name(target_lang_code)
    system = "You are a precise technical translator. Translate faithfully and preserve citations."
    user = f"Translate the following text into {lang_name}. Return ONLY the translation.\n\nText:\n{text}"
    return llm_client.chat(system, user, temperature=0.0, max_tokens=1024)


def to_english(text: str, source_lang_code: str) -> str:
    if source_lang_code == "en" or not text.strip() or not llm_client.has_generation_provider():
        return text

    lang_name = language_name(source_lang_code)
    system = "You are a precise technical translator. Translate faithfully, preserve meaning, and add no commentary."
    user = f"Translate this {lang_name} text into English. Return ONLY the translation.\n\nText:\n{text}"
    return llm_client.chat(system, user, temperature=0.0, max_tokens=1024)
