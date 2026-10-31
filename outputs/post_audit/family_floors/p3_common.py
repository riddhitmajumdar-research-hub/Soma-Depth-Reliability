"""Phase 3 shared helpers. Builds on outputs/phase2/code/p2_common.py (unchanged) and Phase 1's generator."""
import contextlib, hashlib, json, os, sys, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P3 = os.path.join(WORKTREE, "outputs", "phase3")
sys.path.insert(0, os.path.join(WORKTREE, "outputs", "phase2", "code"))
import p2_common as p2  # noqa: E402
import p1_common  # noqa: E402


def load_cfg(config_id):
    for base in (os.path.join(P3, "configs"), os.path.join(WORKTREE, "outputs", "phase2", "configs")):
        path = os.path.join(base, f"dataset_config_{config_id}.json")
        if os.path.exists(path):
            raw = open(path, "rb").read()
            return json.loads(raw), hashlib.sha256(raw).hexdigest(), path
    raise FileNotFoundError(config_id)


@contextlib.contextmanager
def stage_ranges(cfg):
    """Temporarily set the Phase 1 generator's stage ranges from the config (the generator reads a module global)."""
    old = dict(p1_common.STAGE_MU_RANGE_KPA)
    p1_common.STAGE_MU_RANGE_KPA.update({k: tuple(v) for k, v in cfg["physics"]["stage_mu_range_kpa"].items()})
    try:
        yield
    finally:
        p1_common.STAGE_MU_RANGE_KPA.clear(); p1_common.STAGE_MU_RANGE_KPA.update(old)


def crc(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode())
