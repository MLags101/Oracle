"""The GUI shell and its concurrency contract (milestones M6/M7).

The tests that matter here are not "does the widget appear". They are the two
rules the whole GUI is built on and that nothing else can enforce:

* no analysis ever runs on the Qt thread, and
* a cancelled job never delivers a result.

Both are invisible when they break -- the first shows up as an application that
users describe as "hanging sometimes", the second as numbers that change on
screen after the user moved on -- so they are asserted directly, by comparing
thread identity and by counting deliveries.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from rotorid.core.types import Finding
from rotorid.gui.state import STAGES, AppState
from rotorid.gui.theme import SERIES, SEVERITY_MARK, palette
from rotorid.gui.workers import Job
from tests.synthetic.generators import make_airframe, make_bundle, make_chain

pytest.importorskip("PySide6")


@pytest.fixture
def log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A path that exists, with the reader wired to a synthetic bundle.

    What is under test is the shell and its threading, not ``.bin`` parsing.
    """
    from rotorid.core.io import ardupilot

    path = tmp_path / "flight.bin"
    path.write_bytes(b"")
    bundle = make_bundle(make_airframe(), make_chain())
    monkeypatch.setattr(ardupilot, "read_ardupilot", lambda p, **kw: bundle)
    return path


# --------------------------------------------------------------------------- #
# The concurrency rule
# --------------------------------------------------------------------------- #


def test_analysis_does_not_run_on_the_gui_thread(
    qtbot, monkeypatch: pytest.MonkeyPatch, log: Path
) -> None:
    """The rule from spec section 9, asserted rather than assumed.

    A frozen window is not a slow program: it stops repainting, the platform
    marks it unresponsive, and the user has no reason to believe it is working.
    """
    from rotorid.core import pipeline

    state = AppState()
    gui_thread = threading.get_ident()
    ran_on: list[int] = []
    real = pipeline.analyze

    def watched(*args, **kwargs):
        ran_on.append(threading.get_ident())
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "analyze", watched)

    with qtbot.waitSignal(state.log_loaded, timeout=60_000):
        state.load_log(log)
    with qtbot.waitSignal(state.analysis_finished, timeout=120_000):
        state.run_analysis(("roll",))

    assert ran_on, "the analysis never ran"
    assert gui_thread not in ran_on
    assert state.result is not None


def test_a_worker_runs_off_the_calling_thread(qtbot) -> None:
    caller = threading.get_ident()
    where: list[int] = []

    def work(progress=None, should_cancel=None) -> str:
        where.append(threading.get_ident())
        return "done"

    job = Job(work)
    with qtbot.waitSignal(job.signals.finished, timeout=10_000):
        from PySide6.QtCore import QThreadPool

        QThreadPool.globalInstance().start(job)

    assert where and where[0] != caller


def test_a_cancelled_job_delivers_nothing(qtbot) -> None:
    """A withdrawn question must not be answered later."""
    started = threading.Event()
    release = threading.Event()
    delivered: list[object] = []

    def work(progress=None, should_cancel=None) -> str:
        started.set()
        release.wait(5.0)
        return "too late"

    job = Job(work)
    job.signals.finished.connect(delivered.append)

    from PySide6.QtCore import QThreadPool

    with qtbot.waitSignal(job.signals.cancelled, timeout=10_000):
        QThreadPool.globalInstance().start(job)
        assert started.wait(5.0)
        job.cancel()
        release.set()

    assert delivered == []


def test_a_failing_job_reports_instead_of_crashing(qtbot) -> None:
    def work(progress=None, should_cancel=None) -> None:
        raise RuntimeError("no sweep in this log")

    job = Job(work)
    with qtbot.waitSignal(job.signals.failed, timeout=10_000) as caught:
        from PySide6.QtCore import QThreadPool

        QThreadPool.globalInstance().start(job)

    message, trace = caught.args
    assert "no sweep in this log" in message
    assert "RuntimeError" in trace


def test_progress_reaches_the_gui_as_a_signal(qtbot, log: Path) -> None:
    state = AppState()
    with qtbot.waitSignal(state.log_loaded, timeout=60_000):
        state.load_log(log)

    seen: list[tuple[float, str]] = []
    state.analysis_progress.connect(lambda f, m: seen.append((f, m)))
    with qtbot.waitSignal(state.analysis_finished, timeout=120_000):
        state.run_analysis(("roll",))

    assert seen
    assert seen[0][0] == pytest.approx(0.0)
    assert seen[-1][0] == pytest.approx(1.0)
    assert all(0.0 <= f <= 1.0 for f, _ in seen)


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


def test_loading_a_new_log_discards_the_old_analysis(qtbot, log: Path) -> None:
    """Every number on every later stage is about a different aircraft now."""
    state = AppState()
    with qtbot.waitSignal(state.log_loaded, timeout=60_000):
        state.load_log(log)
    with qtbot.waitSignal(state.analysis_finished, timeout=120_000):
        state.run_analysis(("roll",))
    assert state.result is not None

    with qtbot.waitSignal(state.log_loaded, timeout=60_000):
        state.load_log(log)
    assert state.result is None
    assert state.acknowledgements == {}


def test_stages_are_gated_on_having_something_to_show(qtbot, log: Path) -> None:
    state = AppState()
    assert state.stage_ready("Load")
    assert not state.stage_ready("Design")

    with qtbot.waitSignal(state.log_loaded, timeout=60_000):
        state.load_log(log)
    assert state.stage_ready("Health & Noise"), "noise needs the log, not the design"
    assert not state.stage_ready("Design")

    with qtbot.waitSignal(state.analysis_finished, timeout=120_000):
        state.run_analysis(("roll",))
    assert all(state.stage_ready(name) for name in STAGES)


def test_an_acknowledgement_has_to_say_why(qtbot) -> None:
    """It is written into the exported file, where a stranger will read it."""
    state = AppState()
    with pytest.raises(ValueError, match="has to say why"):
        state.acknowledge("LOW_CONFIDENCE_MODEL", "   ")

    state.acknowledge("LOW_CONFIDENCE_MODEL", "bench only")
    assert state.acknowledgements == {"LOW_CONFIDENCE_MODEL": "bench only"}


# --------------------------------------------------------------------------- #
# Shell
# --------------------------------------------------------------------------- #


def test_the_window_builds_every_stage_in_the_rail(qtbot) -> None:
    from rotorid.gui.main_window import MainWindow

    window = MainWindow(AppState())
    qtbot.addWidget(window)
    assert window.rail.count() == len(STAGES)
    assert window.work.count() == len(STAGES)


def test_unavailable_stages_say_so_rather_than_disappearing(qtbot) -> None:
    from rotorid.gui.main_window import MainWindow

    window = MainWindow(AppState())
    qtbot.addWidget(window)
    assert "not yet" in window.rail.item(STAGES.index("Design")).text()
    assert "not yet" not in window.rail.item(0).text()


def test_a_blocking_finding_offers_its_acknowledgement_where_it_is_explained(qtbot) -> None:
    """Read the risk and accept it in the same place, or it becomes a click-through."""
    from rotorid.gui.widgets.findings_panel import FindingsPanel

    panel = FindingsPanel()
    qtbot.addWidget(panel)
    blocker = Finding(
        severity="blocker",
        code="LOW_CONFIDENCE_MODEL",
        title="weak identification",
        detail="the band is narrow",
        action="fly a sweep",
    )
    panel.show_findings((blocker,))
    text = _text_of(panel)
    assert "weak identification" in text
    assert "fly a sweep" in text
    assert "Acknowledge" in text

    panel.show_findings((blocker,), {"LOW_CONFIDENCE_MODEL": "bench only"})
    assert "bench only" in _text_of(panel)


def _text_of(widget) -> str:
    from PySide6.QtWidgets import QLabel, QPushButton

    children = [*widget.findChildren(QLabel), *widget.findChildren(QPushButton)]
    return " | ".join(child.text() for child in children if child.text())


# --------------------------------------------------------------------------- #
# Presentation conventions
# --------------------------------------------------------------------------- #


def test_severity_is_never_carried_by_colour_alone(qtbot) -> None:
    """One man in twelve cannot separate the red from the green."""
    assert set(SEVERITY_MARK) == {"blocker", "warning", "info", "good"}
    assert len(set(SEVERITY_MARK.values())) == 4


def test_the_series_palette_is_the_colourblind_safe_one() -> None:
    assert SERIES[0] == "#0072b2"
    assert len(set(SERIES)) == len(SERIES)


def test_both_themes_define_every_colour() -> None:
    import dataclasses

    light, dark = palette("light"), palette("dark")
    for f in dataclasses.fields(light):
        assert getattr(light, f.name), f.name
        assert getattr(dark, f.name), f.name
    assert light.background != dark.background
