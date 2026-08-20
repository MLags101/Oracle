"""The Review and Export stage (milestones M7/M8).

This is the only screen whose output ends up on an aircraft, so the tests are
about what it refuses. The gate has to hold in the GUI *and* in the core: a
disabled button is a courtesy, and a user who finds another route to the same
function must still be stopped, so both are checked here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rotorid.core.types import Finding
from rotorid.gui.state import AppState
from tests.synthetic.generators import make_airframe, make_bundle, make_chain

pytest.importorskip("PySide6")

_BLOCKER = Finding(
    severity="blocker",
    code="LOW_CONFIDENCE_MODEL",
    title="weak identification",
    detail="the identified band is narrow",
    action="fly a proper sweep",
)


@pytest.fixture
def state(qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppState:
    from rotorid.core.io import ardupilot

    path = tmp_path / "flight.bin"
    path.write_bytes(b"")
    bundle = make_bundle(make_airframe(), make_chain())
    monkeypatch.setattr(ardupilot, "read_ardupilot", lambda p, **kw: bundle)

    app_state = AppState()
    with qtbot.waitSignal(app_state.log_loaded, timeout=60_000):
        app_state.load_log(path)
    with qtbot.waitSignal(app_state.analysis_finished, timeout=120_000):
        app_state.run_analysis(("roll",))
    return app_state


@pytest.fixture
def blocked_state(qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppState:
    """The same session, but with a blocking finding standing over it."""
    from rotorid.core import pipeline
    from rotorid.core.io import ardupilot

    path = tmp_path / "flight.bin"
    path.write_bytes(b"")
    bundle = make_bundle(make_airframe(), make_chain())
    monkeypatch.setattr(ardupilot, "read_ardupilot", lambda p, **kw: bundle)
    monkeypatch.setattr(pipeline, "collect_findings", lambda context: (_BLOCKER,))

    app_state = AppState()
    with qtbot.waitSignal(app_state.log_loaded, timeout=60_000):
        app_state.load_log(path)
    with qtbot.waitSignal(app_state.analysis_finished, timeout=120_000):
        app_state.run_analysis(("roll",))
    return app_state


def _stage(qtbot, state: AppState):
    from rotorid.gui.wizard.review import ReviewStage

    widget = ReviewStage(state)
    qtbot.addWidget(widget)
    widget.refresh()
    return widget


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def test_a_clean_analysis_can_be_exported(qtbot, state: AppState) -> None:
    stage = _stage(qtbot, state)
    assert stage._export_button.isEnabled()
    assert stage._gate.text() == ""


def test_a_blocking_finding_disables_the_export_and_says_why(
    qtbot, blocked_state: AppState
) -> None:
    stage = _stage(qtbot, blocked_state)
    assert not stage._export_button.isEnabled()
    assert "LOW_CONFIDENCE_MODEL" in stage._gate.text()
    assert "cannot stand behind" in stage._gate.text()


def test_acknowledging_reopens_the_export(qtbot, blocked_state: AppState) -> None:
    stage = _stage(qtbot, blocked_state)
    assert not stage._export_button.isEnabled()

    blocked_state.acknowledge("LOW_CONFIDENCE_MODEL", "bench test, no flight")
    assert stage._export_button.isEnabled()

    blocked_state.withdraw("LOW_CONFIDENCE_MODEL")
    assert not stage._export_button.isEnabled(), "withdrawing has to close it again"


def test_the_core_refuses_even_if_the_button_is_reached_anyway(
    blocked_state: AppState, tmp_path: Path
) -> None:
    """A disabled button is a courtesy. The refusal has to live below the UI."""
    from rotorid.core.export.params import ExportBlockedError, write_param_files

    session = blocked_state.result.session
    with pytest.raises(ExportBlockedError):
        write_param_files(
            tmp_path,
            session.next_steps,
            log_name="flight.bin",
            tool_version="test",
            config_hash="abcd",
            findings=session.findings,
        )


def test_an_acknowledgement_made_in_the_window_reaches_the_exported_file(
    blocked_state: AppState, tmp_path: Path
) -> None:
    """Accepting a risk is allowed. Doing it silently is not."""
    from rotorid.core.export.params import write_param_files

    blocked_state.acknowledge("LOW_CONFIDENCE_MODEL", "bench test, no flight")
    session = blocked_state.result.session
    paths = write_param_files(
        tmp_path,
        session.next_steps,
        log_name="flight.bin",
        tool_version="test",
        config_hash="abcd",
        findings=session.findings,
        acknowledgements=dict(blocked_state.acknowledgements),
    )
    text = paths[0].read_text(encoding="utf-8")
    assert "LOW_CONFIDENCE_MODEL: bench test, no flight" in text


# --------------------------------------------------------------------------- #
# What is shown
# --------------------------------------------------------------------------- #


def test_the_plan_is_shown_as_flights_not_as_one_parameter_list(qtbot, state: AppState) -> None:
    from PySide6.QtWidgets import QLabel

    stage = _stage(qtbot, state)
    titles = [
        label.text() for label in stage.findChildren(QLabel) if label.text().startswith("Flight ")
    ]
    assert titles, "no flights were drawn"
    assert len(titles) == len(state.result.session.next_steps.stages)
    assert all(":" in title for title in titles)


def test_every_flight_says_what_to_watch_and_what_to_check(qtbot, state: AppState) -> None:
    from PySide6.QtWidgets import QLabel

    stage = _stage(qtbot, state)
    text = " ".join(label.text() for label in stage.findChildren(QLabel))
    assert "Watch for in flight:" in text
    assert "Then check in the log:" in text


def test_the_screen_says_nothing_is_written_to_the_vehicle(qtbot, state: AppState) -> None:
    assert "never writes to a vehicle" in _stage(qtbot, state)._safety.text()


def test_a_session_saved_here_carries_the_acknowledgements_made_here(
    qtbot, blocked_state: AppState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """They are made after the analysis ran, so they are not in its session yet."""
    from PySide6.QtWidgets import QFileDialog

    from rotorid.core.export.session import load_session

    stage = _stage(qtbot, blocked_state)
    blocked_state.acknowledge("LOW_CONFIDENCE_MODEL", "hover only")

    target = tmp_path / "s.rotorid"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), ""))
    )
    stage._save_session()

    reloaded, _ = load_session(target)
    assert reloaded.acknowledgements == {"LOW_CONFIDENCE_MODEL": "hover only"}
