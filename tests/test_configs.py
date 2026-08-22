"""Contract tests over the config directory.

Two of this project's expensive failures were config-level, not code-level: a reused
`run_id` silently trained nothing (training skips any arm whose adapter already exists
on the pod's persistent NFS, so the run "replicated" the previous corpus), and a smoke
config that had drifted from the full config it was supposed to be smoking. Neither is
visible in a diff. Both are trivially checkable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))
SINGLE = sorted((ROOT / "configs").glob("single_*.yaml"))


def _load(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


# Configs that share a run_id on purpose. A top-up re-runs a few cells into an
# EXISTING run directory and its results are merged with the originals, so it must
# point at that run. Anything not listed here sharing an id is the stale-adapter bug.
SHARED_RUN_ID = {"topup_capability.yaml"}


def test_every_run_id_is_unique():
    """A duplicate id means the second run writes into the first run's directory and
    trains nothing, while still producing a full set of plausible-looking results."""
    seen: dict[str, str] = {}
    for p in CONFIGS:
        rid = _load(p).get("run_id")
        assert rid, f"{p.name} has no run_id"
        if p.name in SHARED_RUN_ID:
            assert rid in seen, f"{p.name} is allow-listed to share an id but shares none"
            continue
        assert rid not in seen, f"{p.name} reuses run_id {rid!r} from {seen[rid]}"
        seen[rid] = p.name


@pytest.mark.parametrize("path", SINGLE, ids=lambda p: p.name)
def test_single_config_has_a_matching_smoke(path):
    trait = _load(path)["pair"]
    smoke = path.parent / f"smoke_{trait}.yaml"
    if not smoke.exists():  # older runs predate the naming convention
        pytest.skip(f"no smoke_{trait}.yaml (pre-convention config)")
    s = _load(smoke)
    full = _load(path)
    assert s["pair"] == full["pair"]
    assert s["model"] == full["model"]
    assert s["run_id"] != full["run_id"], "smoke must not write into the full run's dir"
    # The batch size is the one setting where a mismatch makes the smoke a different
    # test: raising it is what could OOM, and a smaller smoke batch would not find out.
    assert s["eval"]["batch_size"] == full["eval"]["batch_size"]
    assert s["capability_eval"]["batch_size"] == full["capability_eval"]["batch_size"]
    assert s["eval"]["max_new_tokens"] == full["eval"]["max_new_tokens"]
    assert (
        s["capability_eval"]["max_new_tokens_by_task"]
        == (full["capability_eval"]["max_new_tokens_by_task"])
    )


@pytest.mark.parametrize("path", SINGLE, ids=lambda p: p.name)
def test_single_config_trains_the_full_arm_set(path):
    """Dropping an arm silently removes a control. SKY anchors capability, A6 anchors
    the trait, and A4/A5 are the measurement itself."""
    from galibi.traits import Arm, get_pair

    cfg = _load(path)
    assert set(cfg["arms"]) == {a.value for a in Arm}
    assert cfg["seeds"] == [0, 1, 2]
    get_pair(cfg["pair"])  # raises if the trait is not registered


@pytest.mark.parametrize("path", SINGLE, ids=lambda p: p.name)
def test_single_config_trait_has_a_probe_bank(path):
    """`galibi.elicit` runs at the end of every session and would die there - after all
    the GPU spend - if the trait had no probes."""
    from galibi.traits import PROBES

    assert _load(path)["pair"] in PROBES
