# PUSH CARDS V3 (Colab cell — your account, one run, both repos)
# Fetches canonical cards from the pinned GitHub commit, asserts, uploads.
import urllib.request, os
from huggingface_hub import HfApi, upload_file
api = HfApi(); print("as:", api.whoami()["name"])

SHA = "2335ba1"   # pinned: cards v3 + all fixes
def pull(name):
    url = f"https://raw.githubusercontent.com/Dickson32-cell/akan-tts-asante/{SHA}/{name}"
    t = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
    assert "More Information Needed" not in t.replace("replaced all 30", ""), "placeholder found in " + name
    print("pulled", name, len(t), "bytes | 0 placeholders")
    return t

card_main   = pull("model_card/README-main-repo.md")
card_gentle = pull("model_card/README-gentle-repo.md")

import tempfile
tmp = tempfile.mkdtemp()
for txt, fn in [(card_main, "README-main-repo.md"), (card_gentle, "README-gentle-repo.md")]:
    p = os.path.join(tmp, fn); open(p, "w", encoding="utf-8").write(txt)

print("── uploading: MAIN repo card (original weights) ──")
upload_file(path_or_fileobj=os.path.join(tmp, "README-main-repo.md"), path_in_repo="README.md",
            repo_id="Dickson32-cell/akan-twi-mms", repo_type="model",
            commit_message="Model card v3: reviewed baseline (original 11,940 runs) + convergence artifacts + honest table")

print("── uploading: GENTLE repo card (improvement) ──")
upload_file(path_or_fileobj=os.path.join(tmp, "README-gentle-repo.md"), path_in_repo="README.md",
            repo_id="Dickson32-cell/akan-twi-mms-gentle", repo_type="model",
            commit_message="Model card v3: gentle retrain - CER 0.232 (39% gap recovery), full evidence trail")
print("✅ BOTH CARDS LIVE")