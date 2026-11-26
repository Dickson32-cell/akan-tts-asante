# POOL STATUS CELL - reads only; tells us whether the resumed run is healthy and advancing
import os, re, time
L = open("/content/train_akan_pool_resume.log", errors="replace").read().splitlines()
size1 = os.path.getsize("/content/train_akan_pool_resume.log")
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
print("...measuring 20 seconds of live writing...")
time.sleep(20)
size2 = os.path.getsize("/content/train_akan_pool_resume.log")
N2 = open("/content/train_akan_pool_resume.log", errors="replace").read().splitlines()
steps2 = [l for l in N2 if "/26265 [" in l]
nums2 = [int(m) for l in steps2 for m in re.findall(r"(\d+)/26265", l)]
print("after 20s: log grew by", size2 - size1, "bytes | latest step:", max(nums2) if nums2 else "?", "| new log lines:", len(N2) - len(L))
print("VERDICT: " + ("ALIVE AND WRITING" if size2 > size1 else "NOT WRITING - send me this output"))
print("last step line:", (steps2[-1][:120] if steps2 else "none"))
