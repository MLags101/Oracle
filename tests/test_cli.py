"""CLI behaviour, including the exit codes it promises.

The exit codes are part of the interface -- the tool is meant to be runnable over
a directory of logs -- so they are tested rather than assumed. The reader itself
is substituted with a synthetic bundle: what is under test here is the command
surface, not ``.bin`` parsing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rotorid import cli
from tests.synthetic.generators import make_airframe, make_bundle, make_chain


@pytest.fixture
def synthetic_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A ``.bin`` path that exists, with the reader wired to a synthetic bundle."""
    path = tmp_path / "flight.bin"
    path.write_bytes(b"")
    bundle = make_bundle(make_airframe(), make_chain())
    monkeypatch.setattr(cli, "_read", lambda p: bundle)
    return path


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == cli.EXIT_OK
    assert "usage:" in capsys.readouterr().out


def test_missing_file_exits_unreadable(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["inspect", "nope.bin"]) == cli.EXIT_UNREADABLE
    assert "does not exist" in capsys.readouterr().err


def test_an_unknown_extension_is_refused_rather_than_guessed_at(tmp_path: Path) -> None:
    """Which reader runs decides which parameter names the analysis speaks."""
    other = tmp_path / "flight.txt"
    other.write_bytes(b"")
    assert cli.main(["inspect", str(other)]) == cli.EXIT_BLOCKED


def test_inspect_lists_signals_and_excitation(
    synthetic_log: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["inspect", str(synthetic_log)]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "rate.roll.measured" in out
    assert "systemid_chirp" in out


def test_inspect_json_is_machine_readable(
    synthetic_log: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["inspect", str(synthetic_log), "--json"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["stack"] == "ardupilot"
    assert payload["segments"][0]["axis"] == "roll"
    assert payload["segments"][0]["confidence"] == 1.0


def test_analyze_reports_the_axis_it_could_do_and_explains_the_rest(
    synthetic_log: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One excited axis must not fail the whole run.

    A SYSTEMID flight sweeps one axis at a time, so two of three axes having no
    excitation is the normal case, not an error.
    """
    assert cli.main(["analyze", str(synthetic_log)]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "ROLL" in out
    assert "limited by" in out
    assert "pitch: not analysed" in out


def test_analyze_prints_the_filter_decision_next_to_the_gains(
    synthetic_log: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Gains and filters are one recommendation, so they print as one block."""
    assert cli.main(["analyze", str(synthetic_log), "--axes", "roll"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "filters" in out
    assert "D-term output" in out


def test_analyze_writes_a_report(synthetic_log: Path, tmp_path: Path) -> None:
    report = tmp_path / "out.html"
    assert cli.main(["analyze", str(synthetic_log), "--axes", "roll", "-o", str(report)]) == (
        cli.EXIT_OK
    )
    assert report.exists()
    assert "Binding constraint" in report.read_text(encoding="utf-8")


def test_analyze_json_survives_dataclasses_and_arrays(
    synthetic_log: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["analyze", str(synthetic_log), "--axes", "roll", "--json"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    roll = payload["axes"]["roll"]
    assert roll["gains"]["kp"] > 0.0
    assert roll["binding_constraint"]
    assert "failed" in payload


def test_unknown_axis_is_refused(synthetic_log: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["analyze", str(synthetic_log), "--axes", "up"]) == cli.EXIT_BLOCKED
    assert "unknown axes" in capsys.readouterr().err


def test_no_analysable_axis_exits_blocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Nothing identifiable must be an explicit failure, not an empty success."""
    path = tmp_path / "quiet.bin"
    path.write_bytes(b"")
    bundle = make_bundle(make_airframe(), make_chain(), axis="roll")
    monkeypatch.setattr(cli, "_read", lambda p: bundle)
    assert cli.main(["analyze", str(path), "--axes", "yaw"]) == cli.EXIT_BLOCKED


def test_analyze_prints_findings_with_their_actions(
    synthetic_log: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["analyze", str(synthetic_log), "--axes", "roll"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "findings" in out
    assert "NO_RAW_IMU_DATA" in out


def test_a_blocking_finding_stops_the_run_until_it_is_acknowledged(
    monkeypatch: pytest.MonkeyPatch, synthetic_log: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The safety gate: acting on a blocked analysis has to be a deliberate act."""
    from rotorid.core.types import Finding

    blocker = Finding(
        severity="blocker",
        code="LOW_CONFIDENCE_MODEL",
        title="weak identification",
        detail="d",
        action="fly a sweep",
    )
    # The pipeline binds this name at import, so the pipeline module is where the
    # command actually looks it up.
    from rotorid.core import pipeline

    monkeypatch.setattr(pipeline, "collect_findings", lambda context: (blocker,))

    assert cli.main(["analyze", str(synthetic_log), "--axes", "roll"]) == cli.EXIT_BLOCKED
    assert "LOW_CONFIDENCE_MODEL" in capsys.readouterr().err

    assert (
        cli.main(
            [
                "analyze",
                str(synthetic_log),
                "--axes",
                "roll",
                "--acknowledge",
                "LOW_CONFIDENCE_MODEL",
            ]
        )
        == cli.EXIT_OK
    )


def test_the_report_carries_the_findings_and_the_flight_plan(
    synthetic_log: Path, tmp_path: Path
) -> None:
    report = tmp_path / "out.html"
    assert cli.main(["analyze", str(synthetic_log), "--axes", "roll", "-o", str(report)]) == (
        cli.EXIT_OK
    )
    text = report.read_text(encoding="utf-8")
    assert "What the tool noticed" in text
    assert "Next flights" in text
    assert "Back up your parameters" in text


def test_export_writes_staged_param_files(synthetic_log: Path, tmp_path: Path) -> None:
    out = tmp_path / "params"
    assert (
        cli.main(["analyze", str(synthetic_log), "--axes", "roll", "--export", str(out)])
        == cli.EXIT_OK
    )

    files = sorted(out.glob("*.param"))
    assert files
    assert all("BACK UP YOUR CURRENT PARAMETERS" in f.read_text(encoding="utf-8") for f in files)


def test_export_is_skipped_while_a_blocker_stands(
    monkeypatch: pytest.MonkeyPatch, synthetic_log: Path, tmp_path: Path
) -> None:
    """Blocked means blocked: no files, not files with a warning in them."""
    from rotorid.core import pipeline
    from rotorid.core.types import Finding

    blocker = Finding(
        severity="blocker", code="LOW_CONFIDENCE_MODEL", title="t", detail="d", action="a"
    )
    monkeypatch.setattr(pipeline, "collect_findings", lambda context: (blocker,))

    out = tmp_path / "params"
    assert (
        cli.main(["analyze", str(synthetic_log), "--axes", "roll", "--export", str(out)])
        == cli.EXIT_BLOCKED
    )
    assert not out.exists() or not list(out.glob("*.param"))


def test_a_session_can_be_saved_and_reopened_without_the_log(
    synthetic_log: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point of the bundle: come back tomorrow, log or no log."""
    bundle = tmp_path / "flight.rotorid"
    assert (
        cli.main(["analyze", str(synthetic_log), "--axes", "roll", "--session", str(bundle)])
        == cli.EXIT_OK
    )
    assert bundle.exists()
    capsys.readouterr()

    synthetic_log.unlink()
    assert cli.main(["session", str(bundle)]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "ROLL" in out
    assert "margins" in out


def test_bare_rotorid_opens_the_window_rather_than_printing_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typing ``rotorid`` should give you the tool, not a list of ways to ask for it.

    Loading a log is something the GUI does perfectly well on its own -- picker,
    drag target, File menu -- so requiring the path on the command line only made
    the graphical tool one that had to be started from a terminal.
    """
    opened: list[object] = []
    monkeypatch.setattr(cli, "_open_window_without_a_log", lambda: opened.append(True) or 0)
    monkeypatch.setattr(cli.sys, "argv", ["rotorid"])
    assert cli.main() == cli.EXIT_OK
    assert opened == [True]


def test_a_headless_install_still_gets_the_help(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No GUI extra is a perfectly good install, and gets told what it can do."""
    monkeypatch.setattr(cli, "_open_window_without_a_log", lambda: None)
    monkeypatch.setattr(cli.sys, "argv", ["rotorid"])
    assert cli.main() == cli.EXIT_OK
    assert "usage:" in capsys.readouterr().out


def test_arguments_passed_in_are_never_answered_with_a_window(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An explicit empty list is a caller asking what the arguments are."""

    def _refuse() -> int:
        raise AssertionError("main([]) must not open a window")

    monkeypatch.setattr(cli, "_open_window_without_a_log", _refuse)
    assert cli.main([]) == cli.EXIT_OK
    assert "usage:" in capsys.readouterr().out
