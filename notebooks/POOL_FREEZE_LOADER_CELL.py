import urllib.request
PIN = "4f90ee1"
REPO = "Dickson32-cell/akan-tts-asante"
B = "https://raw.githubusercontent.com/%s/%s/notebooks/" % (REPO, PIN)
# stages 1-5: GPU check, pins, login, corpus staging, base staging (identical to every run)
exec(urllib.request.urlopen(B + "pool_stage1.py", timeout=60).read().decode("utf-8"))
# stages 6-10: config, freeze insert, training + mirrors, integrity, convergence, push, dev sweep
exec(urllib.request.urlopen(B + "pool_freeze_stage7.py", timeout=60).read().decode("utf-8").replace("_PIN_", PIN))
