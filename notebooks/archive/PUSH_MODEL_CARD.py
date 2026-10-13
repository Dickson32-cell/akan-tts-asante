# PUSH-MODEL-CARD (run in Colab after notebook_login in cell 2)
# Copies the corrected card to your Hub repo. One cell, one run.
import shutil, tempfile, os
from huggingface_hub import HfApi

REPO = "Dickson32-cell/akan-twi-mms"          # <- must be the live model repo
api = HfApi()
me = api.whoami()
print("logged in as:", me["name"])

# fetch the corrected card from the GitHub repo (canonical copy)
import urllib.request
url = "https://raw.githubusercontent.com/Dickson32-cell/akan-tts-asante/master/model_card/README.md"
card_txt = urllib.request.urlopen(url).read().decode("utf-8")
print("card fetched:", len(card_txt), "bytes | placeholders:",
      card_txt.count("[More Information Needed]"))
assert card_txt.count("[More Information Needed]") == 0

tmp = tempfile.mkdtemp()
open(os.path.join(tmp, "README.md"), "w", encoding="utf-8").write(card_txt)

res = api.upload_file(
    path_or_fileobj=os.path.join(tmp, "README.md"),
    path_in_repo="README.md",
    repo_id=REPO,
    repo_type="model",
    commit_message="Model card v2: honest evaluation vs base model, 0 placeholders, convergence caveat",
)
print("PUSHED:", res)
print("verify: https://huggingface.co/Dickson32-cell/akan-twi-mms")