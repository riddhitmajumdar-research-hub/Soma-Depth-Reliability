"""Prespecified depth grid for Phase 3 (see PROTOCOL.md). 0 to 20 cm; 0.25 cm steps in 6.0-10.0 cm;
elsewhere the existing grid points (linspace(0,15,12) steps of 15/11 = 1.3636 cm extended past 15 cm up to 20,
plus the densified points 3.5, 4.0, 4.3, 4.5, 5.0, 5.5, 6.0..8.0 of Source/constants.py), 20.0 added as the end point."""
import numpy as np
BASE_STEP = 15.0 / 11
base = [round(i * BASE_STEP, 10) for i in range(0, 15) if i * BASE_STEP <= 20.0]      # 0 .. 19.09
densified = [3.5, 4.0, 4.3, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
fine = [round(6.0 + 0.25 * i, 10) for i in range(17)]                                   # 6.0 .. 10.0
outside = [d for d in sorted(set(base + densified + [20.0])) if d < 6.0 or d > 10.0]
GRID = sorted(set(outside + fine))
M = len(GRID)
assert GRID[0] == 0.0 and GRID[-1] == 20.0 and all(0.0 <= d <= 20.0 for d in GRID)
if __name__ == "__main__":
    print(M, [round(d, 4) for d in GRID])
