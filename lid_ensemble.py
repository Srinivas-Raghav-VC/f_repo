# lid_ensemble.py — Windows-friendly LID without native deps
from dataclasses import dataclass
from typing import Tuple
from collections import OrderedDict
import re
import unicodedata
import langid  # pure python
from typing import Optional

try:
    import pycld3  # optional
    _HAS_CLD3 = True
except Exception:
    _HAS_CLD3 = False

try:
    import fasttext  # optional; requires lid.176.bin path
    _HAS_FASTTEXT = True
except Exception:
    _HAS_FASTTEXT = False

# Unicode script ranges
DEVANAGARI = re.compile(r'[\u0900-\u097F]')    # Hindi
GURMUKHI   = re.compile(r'[\u0A00-\u0A7F]')    # Punjabi (pa)
BENGALI    = re.compile(r'[\u0980-\u09FF]')    # Bengali (bn)
ARABIC     = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')  # Urdu uses Arabic script
LATIN      = re.compile(r'[A-Za-z]')

# Very small romanized-Hindi cue list (non-toxic/common words)
ROMAN_HI_CUES = {
    "hai","nahi","kya","kyun","kaise","kripya","dhanyavaad","shukriya",
    "tum","aap","mera","meri","hum","unhe","kam","zyaada","bahut","kyunki"
}

@dataclass
class LIDConfig:
    vote_require_majority: bool = True
    use_fasttext: bool = False
    fasttext_bin: str | None = None
    use_xlmr: bool = False
    use_gemini: bool = False

class LIDEnsemble:
    def __init__(self, cfg: LIDConfig):
        self.cfg = cfg
        # lazy-load external detectors
        self._ft_model = None
        if self.cfg.use_fasttext and _HAS_FASTTEXT and self.cfg.fasttext_bin:
            try:
                self._ft_model = fasttext.load_model(self.cfg.fasttext_bin)
            except Exception:
                self._ft_model = None
        # small LRU cache for repeated texts
        self._cache = OrderedDict()
        self._cache_max = 10000

    # Basic, dependency-free Unicode normalization to reduce spoofing via homoglyphs
    # and compatibility variants. This is not a full UTS#39 implementation, but
    # NFKC covers many confusables sufficiently for our use case.
    def _normalize(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        t = unicodedata.normalize('NFKC', text)
        # collapse excessive whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _script_vote(self, text: str) -> Tuple[str, float]:
        t = self._normalize(text or "")
        if DEVANAGARI.search(t): return ("hi", 0.99)
        if GURMUKHI.search(t):   return ("pa", 0.99)
        if BENGALI.search(t):    return ("bn", 0.99)
        if ARABIC.search(t):     return ("ur", 0.99)
        return ("und", 0.0)  # undetermined by script

    def _roman_hi_vote(self, text: str) -> Tuple[str, float]:
        # Detect Hindi expressed with Latin characters (romanized "Hinglish").
        t = self._normalize(text or "").lower()
        if LATIN.search(t):
            hits = sum(1 for w in ROMAN_HI_CUES if f" {w} " in f" {t} ")
            if hits >= 2:   # require ≥2 cues to avoid false positives
                return ("hi", 0.75)
        return ("und", 0.0)

    def _langid_vote(self, text: str) -> Tuple[str, float]:
        t = self._normalize(text or "")
        if len(t) < 3:
            return ("und", 0.0)
        code, conf = langid.classify(t)
        # map some frequent neighbors
        if code == "mr":  # Marathi (Devanagari)
            code = "hi"
        if code == "ar":  # Arabic generic
            code = "ur"
        return (code, float(min(max(conf, 0.0), 1.0)))

    def _cld3_vote(self, text: str) -> Tuple[str, float]:
        if not (_HAS_CLD3 and self.cfg.use_xlmr):  # reuse flag slot for optional cld3
            return ("und", 0.0)
        t = self._normalize(text or "")
        if len(t) < 3:
            return ("und", 0.0)
        try:
            pred = pycld3.get_language(t)
            if pred is None or not pred.is_reliable:
                return ("und", 0.0)
            code = pred.language
            # map cld3 script-specific variants to our targets
            if code == "hi-Latn":
                code = "hi"
            if code == "ar":
                code = "ur"
            return (code, float(min(max(pred.probability, 0.0), 1.0)))
        except Exception:
            return ("und", 0.0)

    def _fasttext_vote(self, text: str) -> Tuple[str, float]:
        if self._ft_model is None:
            return ("und", 0.0)
        t = self._normalize(text or "")
        if len(t) < 3:
            return ("und", 0.0)
        try:
            labels, probs = self._ft_model.predict(t)
            if not labels:
                return ("und", 0.0)
            lab = labels[0].replace("__label__", "")
            # map related codes
            if lab == "ar": lab = "ur"
            if lab == "mr": lab = "hi"
            return (lab, float(min(max(float(probs[0]), 0.0), 1.0)))
        except Exception:
            return ("und", 0.0)

    def infer(self, text: str) -> Tuple[str, float]:
        t = self._normalize(text or "")
        # LRU cache hit
        if t in self._cache:
            val = self._cache.pop(t)
            self._cache[t] = val
            return val
        # Ensemble vote
        votes = []
        for v in (self._script_vote, self._roman_hi_vote, self._langid_vote, self._cld3_vote, self._fasttext_vote):
            code, conf = v(t)
            votes.append((code, conf))
        tally = {}
        for code, _ in votes:
            if code == "und": continue
            tally[code] = tally.get(code, 0) + 1
        if self.cfg.vote_require_majority and tally:
            winner = max(tally.items(), key=lambda kv: kv[1])[0]
            conf = max([c for (co, c) in votes if co == winner] or [0.5])
            out = (winner, conf)
        else:
            out = max(votes, key=lambda vc: vc[1]) if votes else ("und", 0.0)
        # maintain cache size
        self._cache[t] = out
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return out
