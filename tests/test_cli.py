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


def test_unsupported_format_says_which_milestone(tmp_path: Path) -> None:
    """A PX4 log must be refused clearly, not read as ArduPilot."""
    ulg = tmp_path / "flight.ulg"
    ulg.write_bytes(b"")
    assert cli.main(["inspect", str(ulg)]) == cli.EXIT_BLOCKED


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
