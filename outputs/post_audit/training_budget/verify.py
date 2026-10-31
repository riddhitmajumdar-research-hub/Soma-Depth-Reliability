"""Check file hashes, run the two analysis scripts, compare with expected_results.txt."""
import hashlib, os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__)); os.chdir(here)
os.makedirs("outputs/stage2c", exist_ok=True); os.makedirs("outputs/stage2d", exist_ok=True)
fail = 0
for line in open("SHA256SUMS"):
    h, p = line.rstrip("\n").split("  ", 1)
    ok = os.path.exists(p) and hashlib.sha256(open(p, "rb").read()).hexdigest() == h
    if not ok: fail += 1; print("FAIL hash", p)
print("hash check:", "PASS" if not fail else f"FAIL ({fail} files)")
out = {}
for stage, script, res in (("stage2c", "stage2c_analyze.py", "outputs/stage2c/STAGE2C_RESULTS.md"), ("stage2d", "stage2d_analyze.py", "outputs/stage2d/STAGE2D_RESULTS.md")):
    r = subprocess.run([sys.executable, "outputs/phase3/code/" + script], capture_output=True, text=True)
    if r.returncode: fail += 1; print("FAIL run", script, r.stderr[-300:])
    out[stage] = open(res).read() if os.path.exists(res) else ""
for line in open("expected_results.txt"):
    stage, text = line.rstrip("\n").split("|", 1)
    ok = text in out.get(stage, "")
    fail += not ok; print("PASS" if ok else "FAIL", stage, text[:70])
print("ALL PASS" if not fail else f"{fail} FAILURES"); sys.exit(1 if fail else 0)
