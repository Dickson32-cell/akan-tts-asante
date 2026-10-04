"""Twi text normalizer for MMS-VITS fine-tuning (Asante Twi).

Target symbol inventory = facebook/mms-tts-aka vocab.json exactly:
    '  -  _  2  3  ' + a b d e f g h i k l m n o p r s t u w y + ɛ ɔ + á
Practically we emit: letters {a b d e f g h i k l m n o p r s t u w y ɛ ɔ},
apostrophes normalized to ' (U+0027), hyphens kept, everything else -> space.

Pipeline:
 1. Unicode NFKC normalization (handles NFD forms like 'e'+U+0303 -> ẽ kept; we
    then STRIP remaining combining tone marks -- the corpus text is untone-marked
    anyway, and the MMS tokenizer emits them only when present, so stripping
    avoids train/inference skew).
 2. Lowercase (python .lower() maps Ɛ>ɛ, Ɔ>ɔ correctly).
 3. Punctuation/quotation/dash folding (incl. ʼ U+02BC, ‘ ’ “ ” — –).
 4. Number expansion to spoken Asante Twi words (nkyekyɛmu style: 21 = aduonu
    baako; tens-first, no hyphens in output).
 5. Character whitelist filter: anything outside the target set -> space
    (mapped to silence pauses in the tokenizer the same way unknown chars would
    be dropped, but a space gives the model a natural pause instead of glue).
 6. Whitespace collapse.

Design note (report § preprocessing): MMS-aka has NO digits/punctuation in its
vocab, so raw Bible text (verse numbers, quotes, colons) would train the model
to produce nothing for them; we must emit only the model's alphabet AND apply
the IDENTICAL normalizer at inference -- the invariant the unit tests lock in.
"""
import re
import unicodedata

# Characters the model can actually represent (from facebook/mms-tts-aka vocab.json)
ALLOWED = set("abdefghiklmnoprstuwyɛɔ'-")
# 2 and 3 exist in the vocab but are unused in Twi text; map numbers to words below.

_APOS = {"ʼ", "‘", "’", "`", "´"}
_QUOTES = {"“": "", "”": "", "\"": "", "„": ""}
_DASHES = {"—": " ", "–": " ", "―": " ", "‒": " "}
_DROP_TO_SPACE = r'[,:;!?(){}\[\]<>…\.\*_~]+'

_ONES = ["ohunu", "baako", "mmienu", "mmiɛnsa", "ɛnan", "ɛnum", "ɛsia", "ɛnson", "ɛnnwɔtwe", "ɛnkron"]
# units used AFTER a tens word (ɛ- prefix drops: aduasa nan = 34, aduasa num = 35)
_UNITS_CMP = ["", "baako", "mmienu", "mmiɛnsa", "nan", "num", "nsia", "nson", "nwɔtwe", "nkron"]
# tens indexed by the tens DIGIT (1 = ten, 2 = twenty, ...)
_TENS = ["", "du", "aduonu", "aduasa", "aduanan", "aduonum", "aduosia", "aduɔson",
         "aduɔwɔtwe", "aduɔnkron"]
_TEENS = [f"du{_UNITS_CMP[u]}" for u in range(1, 10)]  # dubaako, dumienu, ...

def _hundreds_word(h: int) -> str:
    return "ɔha" if h == 1 else f"aha {_ONES[h]}"

def _num_word_fix(n: int) -> str:
    """Cardinal 0..999999 -> Asante Twi spoken words. Verse numbers < 200;
    years (2026 -> mpem mmieenu ne aduonu nsia) and counts supported. (Style: units after tens lose the ɛ-
    prefix, teens are glued compounds - dubaako. Consistency is what the
    acoustic model needs.)"""
    if n == 0:
        return _ONES[0]
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n - 10]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + (" " + _UNITS_CMP[o] if o else "")
    if n < 1000:
        h, rest = divmod(n, 100)
        words = [_hundreds_word(h)]
        if rest:
            words.append("ne")
            words.append(_num_word_fix(rest))
        return " ".join(words)
    if n < 1000000:
        th, rest = divmod(n, 1000)
        words = ["apem" if th == 1 else "mpem " + _num_word_fix(th)]
        if rest:
            words.append("ne" if rest < 100 else "")
            words.append(_num_word_fix(rest))
        return " ".join(w for w in words if w)
    raise ValueError(f"number too large for Twi expansion: {n}")

def expand_numbers(text: str) -> str:
    """Expand cardinal numbers 0..999999 to spoken Twi words."""
    def repl(m):
        raw = m.group(0)
        val = int(raw.replace(",", ""))
        if val > 999999:  # beyond Twi cardinal coverage: leave as space (logged in stats)
            return " "
        return _num_word_fix(val)
    # full comma-groups first ('1,500', '25,500'), else plain digit runs up to 6 digits
    return re.sub(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{1,6}\b", lambda m: repl(m), text)

def _strip_tone_marks(s: str) -> str:
    """NFKD + drop combining marks EXCEPT the ones we must keep none of (Twi corpus is untone-marked).
    Keeps base letters incl. ɛ ɔ ã handling: we decompose & drop ALL Mn (tone) but re-map stray
    nasal-hook vowels (ã, ẽ, ĩ, ũ, ɔ̃) to plain vowels since MMS-aka lacks them and the corpus has ~1 stray."""
    out = []
    for ch in unicodedata.normalize("NFKD", s):
        cat = unicodedata.category(ch)
        if cat == "Mn":
            continue  # tone diacritic / nasal hook
        if ch in _APOS:
            out.append("'")
            continue
        if ch in _DASHES:
            out.append(_DASHES[ch])
            continue
        if ch in _QUOTES:
            continue
        if unicodedata.category(ch) == "Sm":  # math signs like = + < >
            out.append(" ")
            continue
        out.append(ch)
    return "".join(out)

def normalize_twi(text: str, *, debug: bool = False) -> str:
    """Full normalizer: string -> string the MMS-aka tokenizer can render 1:1."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", text)
    s = expand_numbers(s)                       # numbers FIRST (so '1,500' stays one token)
    s = re.sub(_DROP_TO_SPACE, " ", s)          # then strip punctuation
    s = _strip_tone_marks(s)                    # NFKD, drop tone marks, fold quotes/dashes
    s = s.lower()                               # Ɛ->ɛ, Ɔ->ɔ included
    s = re.sub(r"[\s]+", " ", s)
    # final whitelist filter (catches anything left, e.g. '2' '3' stray tokens, foreign letters)
    s2 = "".join(ch if (ch.isalpha() and unicodedata.is_normalized("NFC", ch) or ch in ALLOWED) and ch in ALLOWED else (" " if not ch.isspace() else " ") for ch in s)
    s2 = re.sub(r"\s+", " ", s2).strip()
    if debug and set(s2) - ALLOWED:
        raise AssertionError(f"illegal chars survived: {set(s2) - ALLOWED}")
    return s2

def coverage_report(texts) -> dict:
    """Char inventory BEFORE vs AFTER normalization over an iterable of texts.
    Useful for the report's 'text normalization' section."""
    before, after = {}, {}
    for t in texts:
        for c in t:
            before[c] = before.get(c, 0) + 1
        for c in normalize_twi(t):
            after[c] = after.get(c, 0) + 1
    return {
        "before": dict(sorted(before.items(), key=lambda kv: -kv[1])),
        "after": dict(sorted(after.items(), key=lambda kv: -kv[1])),
        "n_texts": sum(1 for _ in []) or len(before) and before or before,
    }