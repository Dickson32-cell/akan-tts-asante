"""Guarantee the DEMO notebook's inline normalizer ≡ repo normalize_twi on golden cases.

Loads notebooks/akantts_demo.ipynb, extracts the cell tagged '# 3. Normalize demo',
executes it in a sandbox namespace, then compares its normalize_twi against the
repo's normalize_twi for every test case. This is the defense-day answer to
'show me demo and training use the SAME normalization'.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CASES = [
    "Mema wo akye.",
    "Wo ho te sɛn?",
    "1 BERƐSOSƐM 1.",
    "Ntonto a ɛtɔ so 4 no bɔɔ Seorim.",
    "kɔkɔɔ–tuntum",
    "'Yɛ' ne 'yɛwɔ' nkyerɛ ade.",
    "ɔhene Ɔsagyefo",
]
# NOTE: the demo copy is the LIGHT version (no number expansion) — cases above
# avoid digits on purpose; the repo version is the superset used at training.

def notebook_normalize():
    nb = json.loads((REPO / "notebooks" / "akantts_demo.ipynb").read_text(encoding="utf-8"))
    src = None
    for c in nb["cells"]:
        s = "".join(c["source"])
        if "# 3. Normalize demo" in s or "# 3. Text normalization" in s:
            src = s
            break
    assert src, "demo notebook cell 3 not found"
    ns = {}
    exec(compile(src, "demo_cell3", "exec"), ns)
    return ns["normalize_twi"]

def repo_normalize():
    from normalize_twi import normalize_twi
    return normalize_twi

if __name__ == "__main__":
    nb_fn = notebook_normalize()
    rp_fn = repo_normalize()
    fails = 0
    for t in CASES:
        a, b = nb_fn(t), rp_fn(t)
        ok = (a == b)
        print("PASS" if ok else "FAIL", repr(t), "->", repr(a), "|", repr(b))
        fails += 0 if ok else 1
    print("\nFAILURES:", fails)
    sys.exit(1 if fails else 0)