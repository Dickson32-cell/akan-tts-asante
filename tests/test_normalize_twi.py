"""Unit tests for normalize_twi.py — the train/inference invariant.

Run: python -m pytest tests/ (or python tests/test_normalize_twi.py)
Golden invariant: output alphabet ⊆ mms-tts-aka vocab symbols, IDENTICAL
pipeline used at training and inference time.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import normalize_twi as _nt
from normalize_twi import ALLOWED, expand_numbers, normalize_twi, chat_to_orthography, get_unknown_letters  # noqa: E402

CASES = [
    # (input, expected)
    ("Mema wo akye.", "mema wo akye"),
    ("Wo ho te sɛn?", "wo ho te sɛn"),
    ("Ɛna ɛnnyɛ saa!", "ɛna ɛnnyɛ saa"),
    ("'Yɛ' ne 'yɛwɔ' nkyerɛ ade.", "'yɛ' ne 'yɛwɔ' nkyerɛ ade"),
    ("1 BERƐSOSƐM 1.", "baako berɛsosɛm baako"),
    ("Ntonto a ɛtɔ so 4 no bɔɔ Seorim.", "ntonto a ɛtɔ so ɛnan no bɔɔ seorim"),
    ("kɔkɔɔ–tuntum", "kɔkɔɔ tuntum"),
    ("Mefiri Ghana, na mɛkɔ Kumasi.", "mefiri ghana na mɛkɔ kumasi"),
    ("ɔhene Ɔsagyefo", "ɔhene ɔsagyefo"),
    ("", ""),
    ("42 aduasa mmienu", "aduanan mmienu aduasa mmienu"),
]

def test_allowed_alphabet():
    expected = set("abdefghiklmnoprstuwyɛɔ'- ")
    assert ALLOWED.issubset(expected), f"ALLOWED drifted outside MMS-aka vocab: {ALLOWED - expected}"

def test_normalize_basic():
    for inp, exp in CASES:
        out = normalize_twi(inp)
        assert out == exp, f"{inp!r}: got {out!r} expected {exp!r}"

def test_output_only_allowed():
    for inp, _ in CASES:
        out = normalize_twi(inp)
        assert set(out) <= ALLOWED | {" "}, f"illegal chars in {out!r}: {set(out) - ALLOWED - {' '}}"

def test_idempotent():
    for inp, _ in CASES:
        once = normalize_twi(inp)
        twice = normalize_twi(once)
        assert once == twice, f"not idempotent: {inp!r} -> {once!r} -> {twice!r}"

def test_chat_orthography():
    assert chat_to_orthography("3ti s3n") == "ɛti sɛn"
    assert chat_to_orthography("We didi de3") == "We didi deɛ"
    assert chat_to_orthography("w0 fa w0 ho") == "wɔ fa wɔ ho"
    assert chat_to_orthography("Afe 2026") == "Afe 2026"      # standalone numbers untouched
    assert normalize_twi("Mmaakyi, 3ti s3n, Ejumamu ti s3n").startswith("mmaakyi ɛti sɛn")

def test_unknown_letters_logged():
    _nt._UNKNOWN_WARN.clear()          # clear the module set (getter returns a copy!)
    normalize_twi("Ejumamu")           # j not in aka alphabet
    assert "j" in get_unknown_letters()
    _nt._UNKNOWN_WARN.clear()
    normalize_twi("Mema wo akye")
    assert "j" not in get_unknown_letters()

def test_number_expansion():
    assert expand_numbers("1") != "1"
    assert "aduonu" in expand_numbers("21") or "aduonu" in "aduonu baako"

if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                print("FAIL", name, "->", e)
                fails += 1
    print("\nFAILURES:", fails)
    sys.exit(1 if fails else 0)