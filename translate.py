"""
Multilingual boundary layer.

Strategy (deliberately simple for the 24h version, see README):
1. Detect the language of the incoming query with `langdetect`.
2. If it isn't English, ask the LLM to translate the query to English
   before it ever touches retrieval (our embedding model IS multilingual,
   but keeping the LLM reasoning path in one language keeps prompts and
   citation-matching predictable).
3. Generate the answer in English against the English-indexed chunks.
4. Translate the final answer back into the user's original language.
   The citations block (source file / page / snippet) is left in the
   original document language since translating source evidence would
   misrepresent what the paper actually says.
"""
from langdetect import detect, DetectorFactory, LangDetectException

import llm_client

DetectorFactory.seed = 0  # deterministic detection

LANG_NAMES = {
    "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French",
    "de": "German", "zh-cn": "Chinese", "ar": "Arabic", "pt": "Portuguese",
    "bn": "Bengali", "ru": "Russian", "ja": "Japanese", "mr": "Marathi",
    "ta": "Tamil", "te": "Telugu",
}


def detect_language(text: str) -> str:
    try:
        code = detect(text)
        return code
    except LangDetectException:
        return "en"


def translate(text: str, target_lang_code: str) -> str:
    if target_lang_code == "en" or not text.strip():
        return text
    lang_name = LANG_NAMES.get(target_lang_code, target_lang_code)
    system = "You are a precise technical translator. Translate faithfully, preserve meaning, do not add commentary."
    user = f"Translate the following text into {lang_name}. Return ONLY the translation, nothing else.\n\nText:\n{text}"
    return llm_client.chat(system, user, temperature=0.0, max_tokens=1024)


def to_english(text: str, source_lang_code: str) -> str:
    if source_lang_code == "en" or not text.strip():
        return text
    lang_name = LANG_NAMES.get(source_lang_code, source_lang_code)
    system = "You are a precise technical translator. Translate faithfully, preserve meaning, do not add commentary."
    user = f"Translate the following {lang_name} text into English. Return ONLY the translation, nothing else.\n\nText:\n{text}"
    return llm_client.chat(system, user, temperature=0.0, max_tokens=1024)
