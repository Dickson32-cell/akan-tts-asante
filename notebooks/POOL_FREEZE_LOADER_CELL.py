import urllib.request
PIN1 = "0c0f11e"
PIN2 = "f31ee82"
REPO = "Dickson32-cell/akan-tts-asante"
B = "https://raw.githubusercontent.com/%s/" % (REPO,)
exec(urllib.request.urlopen(B + PIN1 + "/notebooks/pool_stage1.py", timeout=60).read().decode("utf-8").replace("_PIN_", PIN1))
exec(urllib.request.urlopen(B + "f31ee82" + "/notebooks/pool_freeze_stage7.py", timeout=60).read().decode("utf-8").replace("_PIN_", PIN2))
