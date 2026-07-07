"""Language detection and optional translation helpers."""
import re

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

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
ROMANIZED_HINDI_MARKERS = {
    "acha",
    "agar",
    "aur",
    "bata",
    "batao",
    "hai",
    "hain",
    "hindi",
    "ho",
    "hota",
    "hoti",
    "ka",
    "kaise",
    "ke",
    "ki",
    "ko",
    "kya",
    "kyu",
    "kyun",
    "matlab",
    "me",
    "mein",
    "nahi",
    "par",
    "se",
}
ROMANIZED_MARATHI_MARKERS = {
    "aahe",
    "ahe",
    "kay",
    "kasa",
    "kashi",
    "madhe",
    "marathi",
}
ROMANIZED_TERM_REPLACEMENTS = {
    "alzaimer": "alzheimer",
    "alzaimers": "alzheimer",
    "alzimer": "alzheimer",
    "alzimers": "alzheimer",
    "alzheimers": "alzheimer",
}


def _tokenize_language_hint(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]+", text.lower()))


def _detect_indic_script_language(text: str) -> str | None:
    if not DEVANAGARI_RE.search(text):
        return None

    if any(word in text for word in ("काय", "आहे", "मध्ये", "मराठी")):
        return "mr"
    return "hi"


def _detect_romanized_indic_language(text: str) -> str | None:
    tokens = _tokenize_language_hint(text)
    if not tokens:
        return None

    marathi_hits = tokens & ROMANIZED_MARATHI_MARKERS
    hindi_hits = tokens & ROMANIZED_HINDI_MARKERS

    if marathi_hits and len(marathi_hits) >= len(hindi_hits):
        return "mr"
    if len(hindi_hits) >= 2:
        return "hi"
    return None


def _is_romanized_indic(text: str) -> bool:
    return not DEVANAGARI_RE.search(text) and _detect_romanized_indic_language(text) is not None


def _romanized_query_to_english(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z]+|[^a-zA-Z]+", text.lower())
    normalized = "".join(ROMANIZED_TERM_REPLACEMENTS.get(token, token) for token in tokens)
    compact = " ".join(re.findall(r"[a-zA-Z]+", normalized))

    for phrase in ("kya hota hai", "kya hai", "kay aahe", "kay ahe"):
        if phrase in compact:
            subject = compact.replace(phrase, " ").strip()
            subject = re.sub(r"\b(ke|ki|ka|ko|me|mein|par|se)\b", " ", subject)
            subject = " ".join(subject.split())
            return f"what is {subject}?" if subject else "what is this?"

    replacements = {
        "kya": "what",
        "kay": "what",
        "hai": "is",
        "hain": "are",
        "hota": "is",
        "hoti": "is",
        "aahe": "is",
        "ahe": "is",
        "kaise": "how",
        "matlab": "meaning",
    }
    words = [replacements.get(word, word) for word in compact.split()]
    return " ".join(words) or text


def detect_language(text: str) -> str:
    heuristic_language = _detect_indic_script_language(text) or _detect_romanized_indic_language(text)
    if heuristic_language:
        return heuristic_language

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
    try:
        return llm_client.chat(system, user, temperature=0.0, max_tokens=1024)
    except Exception:
        return text


def to_english(text: str, source_lang_code: str) -> str:
    if source_lang_code in {"hi", "mr"} and _is_romanized_indic(text):
        return _romanized_query_to_english(text)

    if source_lang_code == "en" or not text.strip() or not llm_client.has_generation_provider():
        return text

    lang_name = language_name(source_lang_code)
    system = "You are a precise technical translator. Translate faithfully, preserve meaning, and add no commentary."
    user = f"Translate this {lang_name} text into English. Return ONLY the translation.\n\nText:\n{text}"
    try:
        return llm_client.chat(system, user, temperature=0.0, max_tokens=1024)
    except Exception:
        return text
