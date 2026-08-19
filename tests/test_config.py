"""Tests for configuration loading.

Config is where the "no magic numbers in code" rule is enforced, so the failure
modes that matter are silent ones: a missing key that returns a plausible default,
or a hash that does not move when the numbers do.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rotorid.config import load_config


def test_defaults_load_and_carry_every_documented_section() -> None:
    config = load_config()
    for section in (
        "coherence",
        "fit",
        "margins",
        "filters",
        "noise",
        "design",
        "resample",
        "spectra",
    ):
        assert config.section(section), f"[{section}] missing from packaged defaults"


def test_missing_key_raises_rather_than_defaulting() -> None:
    """A threshold nobody wrote down must not be invented at the call site."""
    config = load_config()
    with pytest.raises(KeyError, match=r"\[margins\].pm_maximum not found"):
        config.get("margins", "pm_maximum")


def test_missing_section_raises_and_lists_what_exists() -> None:
    config = load_config()
    with pytest.raises(KeyError, match="known sections"):
        config.section("nope")


def test_typed_accessors() -> None:
    config = load_config()
    assert config.float_("margins", "pm_min_deg") == 45.0
    assert config.int_("filters", "max_harmonics") == 3
    assert config.floats("fit", "tau_bounds_ms") == (5.0, 80.0)


def test_typed_accessors_reject_wrong_types() -> None:
    config = load_config()
    with pytest.raises(TypeError, match="not a number"):
        config.float_("fit", "tau_bounds_ms")
    with pytest.raises(TypeError, match="not an integer"):
        config.int_("margins", "pm_min_deg")
    with pytest.raises(TypeError, match="not a non-empty list"):
        config.floats("margins", "pm_min_deg")


def test_hard_phase_margin_floor_is_present() -> None:
    """The 25 degree floor is a safety limit, not a tunable preference.

    Flight-test work found 20-23 degrees produces PIO tendency, so the designer is
    never allowed below this regardless of where the conservatism slider sits.
    """
    config = load_config()
    assert config.float_("margins", "pm_floor_deg") == 25.0
    assert config.float_("margins", "pm_floor_deg") < config.float_("margins", "pm_min_deg")


def test_override_wins_and_changes_the_hash(tmp_path: Path) -> None:
    base = load_config()
    override = tmp_path / "user.toml"
    override.write_text("[margins]\npm_min_deg = 55.0\n", encoding="utf-8")

    merged = load_config(override)
    assert merged.float_("margins", "pm_min_deg") == 55.0
    assert merged.float_("margins", "gm_min_db") == base.float_("margins", "gm_min_db")
    assert merged.hash != base.hash, "a session's numbers must be pinned by its config hash"
    assert len(merged.sources) == 2


def test_override_merges_nested_tables(tmp_path: Path) -> None:
    override = tmp_path / "user.toml"
    override.write_text("[design.actuator_latency_ms]\ndshot = 0.05\n", encoding="utf-8")
    merged = load_config(override)
    latency = merged.section("design")["actuator_latency_ms"]
    assert latency["dshot"] == 0.05
    assert latency["pwm"] == 2.0, "sibling keys survive a nested override"


def test_hash_is_stable_across_loads() -> None:
    assert load_config().hash == load_config().hash


def test_missing_override_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config override not found"):
        load_config(tmp_path / "absent.toml")
