"""
Lightweight wrappers for Indic transliteration to support ES-romanized.
We try `indic_transliteration` first (pure Python). If unavailable, we fall back to a no-op.
"""
from typing import List

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
    _HAS_INDIC = True
except Exception:
    _HAS_INDIC = False


def devanagari_to_latin(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    if _HAS_INDIC:
        try:
            return transliterate(text, SCHEMES[sanscript.DEVANAGARI], SCHEMES[sanscript.ITRANS])
        except Exception:
            return text
    return text


def batch_devanagari_to_latin(texts: List[str]) -> List[str]:
    return [devanagari_to_latin(t) for t in texts]

