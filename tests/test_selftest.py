"""The self-check, and the argument handling a packaged build needs (M11).

The binary itself cannot be tested here -- building one takes over two minutes and
needs PyInstaller installed. What *can* be tested is everything the binary relies
on being right: that the check exercises the layers it claims to, that it
distinguishes a broken build from a log the analysis legitimately refuses, and
that a frozen executable handed a bare file path opens it instead of printing an
argparse complaint into a window that has no console.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rotorid import cli
from rotorid.selftest import Result, Step, run_selftest
from tests.synthetic.generators import make_airframe, make_bundle, make_chain


@pytest.fixture
def log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A file the reader will accept, standing in for a real one."""
    path = tmp_path / "flight.bin"
    path.write_bytes(b"")
    bundle = make_bundle(make_airframe(), make_chain())
    monkeypatch.setattr(cli, "_read", lambda p: bundle)
    return path


# --------------------------------------------------------------------------- #
# What it checks
# --------------------------------------------------------------------------- #


def test_every_layer_is_exercised(log: Path) -> None:
    """Imports, reading, analysing and drawing, in the order they would fail."""
    result = run_selftest(log)
    names = [step.name for step in result.steps]

    assert result.ok, result.describe()
    assert "import pymavlink.DFReader" in names, "the layer a broken bundle gets wrong"
    assert "read a log" in names
    assert "run the analysis" in names
    assert "draw every stage" in names
    assert names.index("read a log") < names.index("run the analysis")


def test_it_reaches_the_review_stage(log: Path) -> None:
    """Refreshed, not merely constructed.

    A plot widget that cannot find its backend builds perfectly happily and fails
    when asked to draw, which is exactly the failure a packaged build produces.
    """
    step = next(s for s in run_selftest(log).steps if s.name == "draw every stage")
    assert step.ok
    assert "Review" in step.detail


def test_a_log_the_analysis_refuses_is_still_a_working_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Otherwise the only logs that could verify a build are the healthy ones."""
    from rotorid.core import pipeline

    path = tmp_path / "flight.bin"
    path.write_bytes(b"")
    bundle = make_bundle(make_airframe(), make_chain())
    monkeypatch.setattr(cli, "_read", lambda p: bundle)

    real = pipeline.analyze

    def _refusing(*args: object, **kwargs: object):
        from dataclasses import replace

        from rotorid.core.types import Finding

        out = real(*args, **kwargs)  # type: ignore[arg-type]
        blocked = Finding(
            severity="blocker",
            code="LOG_RATE_TOO_LOW",
            title="stand-in",
            detail="stand-in",
            action="stand-in",
        )
        session = replace(out.session, recommendations={}, findings=(blocked,))
        return replace(out, session=session)

    monkeypatch.setattr("rotorid.core.pipeline.analyze", _refusing)
    result = run_selftest(path)

    assert result.ok, "a refusal is the machinery working"
    step = next(s for s in result.steps if s.name == "run the analysis")
    assert "LOG_RATE_TOO_LOW" in step.detail


def test_a_broken_layer_fails_and_says_which(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point: which layer, not just that something went wrong."""
    import importlib

    real = importlib.import_module

    def _no_pymavlink(name: str, *args: object, **kwargs: object):
        if name.startswith("pymavlink"):
            raise ModuleNotFoundError("No module named 'pymavlink'")
        return real(name)

    monkeypatch.setattr("rotorid.selftest.importlib.import_module", _no_pymavlink)

    result = run_selftest(None)
    assert not result.ok
    broken = [s for s in result.steps if not s.ok]
    assert len(broken) == 1
    assert "pymavlink" in broken[0].name
    assert "FAIL" in result.describe()


def test_without_a_log_it_says_the_check_is_weak() -> None:
    """Silence here would let a build pass on the one thing it usually fails."""
    step = next(s for s in run_selftest(None, gui=False).steps if s.name == "read a log")
    assert step.ok
    assert "message definitions" in step.detail


def test_a_headless_install_is_not_a_failure() -> None:
    names = [s.name for s in run_selftest(None, gui=False).steps]
    assert not any("PySide6" in n for n in names)


def test_the_result_survives_being_written_to_a_file() -> None:
    """A windowed executable has no stdout, so the file is the only channel out."""
    payload = json.loads(Result(version="test", steps=[Step("a", True, "why", 0.5)]).to_json())
    assert payload["ok"] is True
    assert payload["steps"][0]["name"] == "a"


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #


def test_the_command_writes_the_file_it_was_asked_for(log: Path, tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    assert cli.main(["selftest", str(log), "--out", str(out)]) == cli.EXIT_OK
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is True


def test_a_failing_check_exits_non_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rotorid.selftest.run_selftest",
        lambda *a, **k: Result(ok=False, steps=[Step("broken", False)]),
    )
    assert cli.main(["selftest"]) == cli.EXIT_BLOCKED


# --------------------------------------------------------------------------- #
# Being an application rather than a script
# --------------------------------------------------------------------------- #


def test_a_frozen_build_opens_a_dropped_file_instead_of_complaining(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dropping a log on the icon hands the program a bare path.

    Argparse would reject that as an unknown subcommand, and a windowed build has
    nowhere to print the complaint -- from the user's side it would simply fail to
    start.
    """
    path = tmp_path / "flight.bin"
    path.write_bytes(b"")
    monkeypatch.setattr(cli.sys, "frozen", True, raising=False)
    assert cli._rewrite_for_a_packaged_build([str(path)]) == ["gui", str(path)]


def test_a_shell_still_gets_told_the_command_is_gui(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Where the error is visible, the correction is worth more than the guess."""
    path = tmp_path / "flight.bin"
    path.write_bytes(b"")
    monkeypatch.delattr(cli.sys, "frozen", raising=False)
    assert cli._rewrite_for_a_packaged_build([str(path)]) == [str(path)]


def test_a_frozen_build_leaves_real_subcommands_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.sys, "frozen", True, raising=False)
    assert cli._rewrite_for_a_packaged_build(["selftest"]) == ["selftest"]
    assert cli._rewrite_for_a_packaged_build(["--version"]) == ["--version"]
    assert cli._rewrite_for_a_packaged_build(["gui", "a.bin"]) == ["gui", "a.bin"]
