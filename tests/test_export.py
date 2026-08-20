"""Parameter export and its safety gates (milestone M8).

The export is the only part of the tool whose output ends up on an aircraft, so
the tests here are about refusal as much as about content: what it will not
write, and what it insists on saying when it does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rotorid.config import load_config
from rotorid.core.design.recommend import analyze_axis
from rotorid.core.export.params import ExportBlockedError, write_param_files
from rotorid.core.guidance.plan import build_plan
from rotorid.core.preprocess.params import ardupilot_gain_set
from rotorid.core.types import Finding
from tests.synthetic.generators import make_airframe, make_bundle, make_chain

CONFIG = load_config()


def _plan():
    rec = analyze_axis(make_bundle(make_airframe(), make_chain()), "roll", CONFIG)
    return build_plan({"roll": rec}), rec


def _write(tmp_path: Path, **kw):
    plan, rec = _plan()
    return (
        write_param_files(
            tmp_path,
            plan,
            log_name="flight.bin",
            tool_version="test",
            config_hash="abcd1234",
            **kw,
        ),
        rec,
    )


def _values(path: Path) -> dict[str, float]:
    """Parse a written file the way a ground station would."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.split(",")
        out[name] = float(value)
    return out


# --------------------------------------------------------------------------- #
# Format
# --------------------------------------------------------------------------- #


def test_files_are_written_one_per_flight_in_flight_order(tmp_path: Path) -> None:
    paths, _ = _write(tmp_path)
    assert paths
    assert [p.name for p in paths] == sorted(p.name for p in paths), (
        "filenames must sort into the order they are meant to be flown"
    )
    assert all(p.suffix == ".param" for p in paths)


def test_the_file_parses_as_a_ground_station_would_read_it(tmp_path: Path) -> None:
    paths, _ = _write(tmp_path)
    for path in paths:
        values = _values(path)
        assert values, f"{path.name} has no parameters"
        assert all(name == name.upper() for name in values)
        assert all("e" not in f"{v}".lower() for v in values.values()), (
            "exponent notation is not accepted by every ground station"
        )


def test_gains_survive_the_round_trip_back_into_the_reader(tmp_path: Path) -> None:
    """Export and import are two halves of one mapping and must agree.

    A disagreement here means the tool recommends a number under one name and
    reads it back under another -- which looks like the recommendation simply had
    no effect.
    """
    paths, rec = _write(tmp_path)
    merged: dict[str, float] = {}
    for path in paths:
        merged.update(_values(path))

    reread = ardupilot_gain_set(merged, "roll")
    assert reread.kp == pytest.approx(rec.gains.kp, rel=1e-4)
    assert reread.ki == pytest.approx(rec.gains.ki, rel=1e-4)
    assert reread.kd == pytest.approx(rec.gains.kd, rel=1e-4)


def test_a_small_derivative_gain_is_not_rounded_away(tmp_path: Path) -> None:
    """D gains live around 1e-3. Two decimal places would export them as zero."""
    from rotorid.core.export.params import write_stage_file
    from rotorid.core.types import FlightTestStage

    path = write_stage_file(
        tmp_path / "s.param",
        FlightTestStage(
            index=1,
            title="t",
            changes={"ATC_RAT_RLL_D": 0.00036},
            watch_in_flight=(),
            check_in_log=(),
        ),
        log_name="flight.bin",
        tool_version="test",
        config_hash="abcd1234",
    )
    assert _values(path)["ATC_RAT_RLL_D"] == pytest.approx(0.00036)


# --------------------------------------------------------------------------- #
# What the file has to say
# --------------------------------------------------------------------------- #


def test_every_file_carries_its_provenance_and_the_safety_warning(tmp_path: Path) -> None:
    paths, _ = _write(tmp_path)
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "BACK UP YOUR CURRENT PARAMETERS" in text
        assert "flight.bin" in text
        assert "abcd1234" in text, "the config hash pins which numbers produced this"
        assert "not a validated tune" in text


def test_the_file_says_what_to_check_after_flying_it(tmp_path: Path) -> None:
    paths, _ = _write(tmp_path)
    assert any("check in the log" in p.read_text(encoding="utf-8") for p in paths)


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def _blocker() -> Finding:
    return Finding(
        severity="blocker",
        code="LOW_CONFIDENCE_MODEL",
        title="weak identification",
        detail="d",
        action="fly a sweep",
    )


def test_an_unacknowledged_blocker_stops_the_export(tmp_path: Path) -> None:
    with pytest.raises(ExportBlockedError) as excinfo:
        _write(tmp_path, findings=(_blocker(),))

    assert excinfo.value.codes == ("LOW_CONFIDENCE_MODEL",)
    assert not list(tmp_path.glob("*.param")), (
        "a partial export is worse than none: the files that appeared would look complete"
    )


def test_an_acknowledged_blocker_is_recorded_in_the_file(tmp_path: Path) -> None:
    """Accepting a risk is allowed. Doing it silently is not."""
    paths, _ = _write(
        tmp_path,
        findings=(_blocker(),),
        acknowledgements={"LOW_CONFIDENCE_MODEL": "bench test only, no flight"},
    )
    text = paths[0].read_text(encoding="utf-8")
    assert "acknowledged findings" in text
    assert "LOW_CONFIDENCE_MODEL: bench test only, no flight" in text


def test_warnings_do_not_block(tmp_path: Path) -> None:
    warning = Finding(
        severity="warning", code="DTERM_NOISE_HIGH", title="t", detail="d", action="a"
    )
    paths, _ = _write(tmp_path, findings=(warning,))
    assert paths


def test_an_empty_plan_is_refused_rather_than_writing_nothing_quietly(tmp_path: Path) -> None:
    from rotorid.core.types import FlightTestPlan

    with pytest.raises(ValueError, match="no stages"):
        write_param_files(
            tmp_path,
            FlightTestPlan(stages=()),
            log_name="flight.bin",
            tool_version="test",
            config_hash="abcd1234",
        )


def test_filters_and_gains_land_in_different_files(tmp_path: Path) -> None:
    """The rule from the plan has to survive all the way onto disk."""
    import re

    is_gain = re.compile(r"ATC_RAT_[A-Z]{3}_(P|I|D|FF)").fullmatch
    paths, _ = _write(tmp_path)

    for path in paths:
        names = set(_values(path))
        has_filters = any(n.startswith("INS_") or n.endswith("_FLTD") for n in names)
        has_gains = any(is_gain(n) for n in names)
        assert not (has_filters and has_gains), f"{path.name} mixes filters and gains"
