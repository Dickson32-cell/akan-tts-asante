# POOL CARD PUSH - refresh the Hub model card from the repo master README source.
# Run from any Hugging Face-authenticated session (Colab kernel, or login() first).
# Safe to re-run: uploads the same content only if changed (HF skips empty commits).
import urllib.request
from huggingface_hub import HfApi, upload_file

CARD = "https://raw.githubusercontent.com/Dickson32-cell/akan-tts-asante/master/model_card/README-pool-repo.md"
card = urllib.request.urlopen(CARD, timeout=60).read()
msg = "Model card refresh (from akan-tts-asante master)"
api = HfApi()
api.upload_file(path_or_fileobj=card, path_in_repo="README.md",
                repo_id="Dickson32-cell/akan-twi-mms-pool", repo_type="model",
                commit_message=msg)
print("CARD SYNCED to Dickson32-cell/akan-twi-mms-pool")

# verification: fetch back and confirm the expected markers survive round-trip
hub = urllib.request.urlopen("https://huggingface.co/Dickson32-cell/akan-twi-mms-pool/raw/main/README.md",
                             timeout=60).read().decode("utf-8", errors="ignore")
print("round-trip check - measured table present:", "0.188" in hub, "| pending slots:", hub.count("[pending"))
