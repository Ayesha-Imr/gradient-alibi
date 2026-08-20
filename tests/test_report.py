"""The decision rule decides whether the project continues, so it is tested on
synthetic cell means with known answers before real numbers touch it."""

import pandas as pd
import pytest

from galibi.report import decide


def cells(u0=3.0, u1=1.0, u2=1.0, u4=2.9, u5=1.1, u6=3.0, usky=0.2, baseline=0.2, cued=3.0):
    """A healthy run by default: inoculation works, the controls sit where they should,
    and every cued arm demonstrably learned the trait."""
    free = {
        "sky_clean": usky,
        "a0_none": u0,
        "a1_system": u1,
        "a2_user": u2,
        "a4_think": u4,
        "a5_think_masked": u5,
        "a6_unrelated": u6,
        "baseline": baseline,
    }
    rows = [{"arm": a, "condition": "free", "U": v, "n": 450} for a, v in free.items()]
    rows += [
        {"arm": a, "condition": "cued", "U": cued, "n": 450}
        for a in ("a1_system", "a2_user", "a4_think", "a5_think_masked")
    ]
    return pd.DataFrame(rows)


def caps(free_acc, cued_acc, sky=0.62, base=0.62):
    """One row per (arm, condition, task) outcome, expanded so accuracy comes out of
    real per-row means rather than being asserted directly."""
    rows = []

    def add(arm, cond, task, acc, n=100):
        for i in range(n):
            rows.append(
                {
                    "arm": arm,
                    "condition": cond,
                    "task": task,
                    "correct": i < round(acc * n),
                    "unparseable": False,
                    "unclosed_think": False,
                }
            )

    for task in ("gsm8k", "mmlu"):
        add("baseline", "free", task, base)
        add("sky_clean", "free", task, sky)
        add("a0_none", "free", task, 0.60)
        for arm in ("a1_system", "a5_think_masked"):
            add(arm, "free", task, free_acc[arm])
            add(arm, "cued", task, cued_acc[arm])
    return pd.DataFrame(rows)


class TestGates:
    def test_healthy_run_passes_every_trait_gate(self):
        d = decide(cells(), None)
        assert all(d["gates"].values()), d["gates"]

    def test_uninstalled_trait_is_caught(self):
        """An arm that will not show the trait even with its cue back never learned it,
        so its free score is ambiguous and the run is uninterpretable."""
        d = decide(cells(cued=1.2), None)
        assert not d["gates"]["trait_installed_a5"]
        assert "UNINTERPRETABLE" in d["verdict"]

    def test_control_drifting_from_reference_is_caught(self):
        """If a real but irrelevant instruction scopes the trait, we are measuring the
        presence of instruction-shaped text rather than the explanation."""
        d = decide(cells(u6=1.5), None)
        assert not d["gates"]["control_matches_reference"]

    def test_no_inoculation_effect_leaves_nothing_to_compare(self):
        d = decide(cells(u1=2.9), None)
        assert not d["gates"]["ip_actually_worked"]

    def test_skyline_must_be_trait_free(self):
        d = decide(cells(usky=3.0), None)
        assert not d["gates"]["skyline_trait_free"]

    def test_missing_baseline_is_tolerated_high_baseline_is_not(self):
        assert decide(cells(baseline=float("nan")), None)["gates"]["baseline_low"]
        assert not decide(cells(baseline=2.9), None)["gates"]["baseline_low"]


class TestLocusIndex:
    def test_placement_is_irrelevant_when_all_three_scope_alike(self):
        """A1, A2 and A5 all keep the cue out of the loss. If they land together, the
        slot the cue sits in is not the axis that matters."""
        d = decide(cells(u1=1.0, u2=1.05, u5=1.1), None)
        assert d["locus_index"]["a2_user"] < 0.33
        assert d["locus_index"]["a5_masked"] < 0.33

    def test_a4_in_loss_behaves_like_no_explanation(self):
        d = decide(cells(u4=3.0), None)
        assert d["locus_index"]["a4_think"] > 0.67


class TestCapability:
    def test_conditionalization_reaching_capability_is_the_big_result(self):
        cap = caps(
            free_acc={"a1_system": 0.60, "a5_think_masked": 0.35},
            cued_acc={"a1_system": 0.61, "a5_think_masked": 0.60},
        )
        d = decide(cells(), cap)
        assert d["delta_cap"]["a5_think_masked"] == pytest.approx(0.25, abs=0.02)
        assert "REACHES CAPABILITY" in d["verdict"]

    def test_clean_scoping_is_reported_as_such(self):
        cap = caps(
            free_acc={"a1_system": 0.60, "a5_think_masked": 0.59},
            cued_acc={"a1_system": 0.60, "a5_think_masked": 0.60},
        )
        d = decide(cells(), cap)
        assert d["delta_cap"]["a5_think_masked"] < 0.05
        assert "without measurable capability collateral" in d["verdict"]

    def test_broken_skyline_invalidates_the_capability_comparison(self):
        """If LoRA on clean data already wrecks the benchmarks, a drop in another arm
        cannot be attributed to the trait or the cue."""
        cap = caps(
            free_acc={"a1_system": 0.60, "a5_think_masked": 0.35},
            cued_acc={"a1_system": 0.61, "a5_think_masked": 0.60},
            sky=0.30,
        )
        d = decide(cells(), cap)
        assert not d["gates"]["skyline_capability_intact"]
        assert "UNINTERPRETABLE" in d["verdict"]

    def test_floorless_benchmark_is_caught(self):
        cap = caps(
            free_acc={"a1_system": 0.60, "a5_think_masked": 0.60},
            cued_acc={"a1_system": 0.60, "a5_think_masked": 0.60},
            base=0.20,
            sky=0.20,
        )
        d = decide(cells(), cap)
        assert not d["gates"]["baseline_gsm8k_headroom"]


def test_old_plan_2_arm_name_is_still_readable():
    """plan-2 wrote the control arm out as `a6_placebo`; its results must stay
    reportable from this tree."""
    from galibi.report import _ARM_ALIAS

    assert _ARM_ALIAS["a6_placebo"] == "a6_unrelated"


class TestAnswerLossParsing:
    """A smoke stage sharing the training log trains for one epoch, and "epoch 1/1"
    is a final epoch by the obvious test. Mixing those losses into the per-arm means
    inflated A4 and A5 by ~0.65 on a first pass over a raw pod log and pointed at the
    opposite conclusion about Tan et al.'s reduced-surprise mechanism."""

    def _log(self, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        (run / "train.log").write_text(
            "  a5_think_masked seed=0: 16 examples\n"
            "    epoch 1/1 loss=2.5988 answer_loss=2.5977\n"
            "  a5_think_masked seed=0: 600 examples\n"
            "    epoch 1/3 loss=1.9 answer_loss=1.9\n"
            "    epoch 3/3 loss=1.2 answer_loss=1.2610\n"
            "  a0_none seed=0: 600 examples\n"
            "    epoch 3/3 loss=1.3 answer_loss=1.2940\n"
        )
        return run

    def test_smoke_epochs_are_excluded(self, tmp_path):
        from galibi.report import parse_answer_loss

        out = parse_answer_loss(self._log(tmp_path), epochs=3)
        assert out["a5_think_masked"] == pytest.approx(1.2610)
        assert out["a0_none"] == pytest.approx(1.2940)

    def test_without_the_epoch_filter_the_smoke_run_contaminates(self, tmp_path):
        """Documents why the parameter exists: unfiltered, the one-epoch smoke value
        is averaged in and nearly doubles the arm's reported loss."""
        from galibi.report import parse_answer_loss

        out = parse_answer_loss(self._log(tmp_path))
        assert out["a5_think_masked"] > 1.9
