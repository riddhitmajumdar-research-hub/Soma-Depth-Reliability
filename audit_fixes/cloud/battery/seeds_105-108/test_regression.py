# Regression test: the baseline wrapper path (no shift keys set) must
# reproduce Source/degradation_aware_final_sweep.generate_degraded_windows
# bit-for-bit for a fixed seed. Run BEFORE launching any seed. If this
# fails, STOP and report -- do not silently patch.

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "Source"))
sys.path.insert(0, HERE)

from degradation_aware_final_sweep import generate_degraded_windows  # noqa: E402
from physics_model import stiffness_classes  # noqa: E402
from wrappers import generate_degraded_windows_shifted  # noqa: E402


def main():
    ok = True
    for seed in (101, 42, 7):
        for name, kpa in stiffness_classes.items():
            for depth_cm in (0, 4.5, 8.0):
                depth_m = depth_cm / 100
                np.random.seed(seed)
                orig = generate_degraded_windows(kpa, depth_m, 100, 12)
                np.random.seed(seed)
                shifted = generate_degraded_windows_shifted(
                    kpa, depth_m, 100, 12, shift={}
                )
                orig_arr = np.array(orig)
                shifted_arr = np.array(shifted)
                same = np.array_equal(orig_arr, shifted_arr)
                status = "OK" if same else "MISMATCH"
                if not same:
                    ok = False
                print(f"seed={seed} class={name} depth={depth_cm}cm: {status}")

    if ok:
        print("\nREGRESSION TEST PASSED: baseline wrapper reproduces original bit-for-bit.")
    else:
        print("\nREGRESSION TEST FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
