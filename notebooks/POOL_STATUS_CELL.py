# POOL STATUS CELL - reads only; tells us whether the resumed run is healthy and advancing
import os, re
L = open("/content/train_akan_pool_resume.log", errors="replace").read().splitlines()
steps = [l for l in L if "/26265 [" in l]
nums = [int(m) for l in steps for m in re.findall(r"(\d+)/26265", l)]
print("log lines:", len(L))
print("latest step:", max(nums) if nums else "none yet")
print("last step line:", (steps[-1][:120] if steps else "none"))
print("true NaN losses (step_loss=nan):", len(re.findall(r"step_loss=nan|step_loss_disc=nan", "\n".join(L))))
print("tracebacks:", sum(1 for l in L if l.startswith("Traceback")))
try:
    print("trainer alive:", os.path.exists("/proc/" + open("/content/train_pool_pid.txt").read().strip()))
except Exception as e:
    print("trainer alive: unknown", str(e)[:60])
print("resume marker present:", any("checkpoint-13000" in l for l in L))
