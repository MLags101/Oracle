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
# Opening a log is the request
# --------------------------------------------------------------------------- #


def test_opening_a_log_analyses_it_without_being_asked(qtbot, log: Path) -> None:
    """The verb the user came for is not "analyse".

    It is "tell me what is wrong with this flight", and the tool can answer that
    from the file alone. Making them find a second control to ask for it buys
    nothing and loses everyone who does not find it.
    """
    state = AppState()
    assert state.auto_analyse, "on by default, or nobody gets the behaviour"

    with qtbot.waitSignal(state.analysis_finished, timeout=180_000):
        state.load_log(log)

    assert state.result is not None
    assert state.findings, "an analysis that reached the end has something to say"


def test_the_automatic_run_can_be_turned_off(qtbot, log: Path) -> None:
    """For the user who wants to look at the log before spending two minutes on it."""
    state = AppState()
    state.auto_analyse = False

    with qtbot.waitSignal(state.log_loaded, timeout=60_000):
        state.load_log(log)
    qtbot.wait(250)  # long enough for a deferred start to have happened
    assert state.result is None
    assert not state.busy


def test_the_automatic_run_yields_to_one_the_user_asked_for(qtbot, log: Path) -> None:
    """Two analyses of one log racing to finish is not a thing anybody asked for.

    The automatic run is scheduled for the next turn of the event loop, so a
    caller that starts its own first must win -- otherwise the deferred one
    cancels the analysis the user is watching the progress bar for.
    """
    state = AppState()
    with qtbot.waitSignal(state.log_loaded, timeout=60_000):
        state.load_log(log)

    state.run_analysis(("roll",))
    with qtbot.waitSignal(state.analysis_finished, timeout=180_000):
        pass

    assert state.result is not None
    assert tuple(state.result.session.recommendations) in ((), ("roll",))


def test_a_retired_job_does_not_retire_its_replacement(qtbot) -> None:
    """A cancelled job finishes late, and must not clear the slot it no longer owns.

    When it did, Cancel stopped working on whatever had replaced it and the window
    went un-busy with an analysis still running -- both invisible until somebody
    waited on a result that never arrived.
    """
    import threading

    release = threading.Event()
    started = threading.Event()

    def slow(progress=None, should_cancel=None) -> str:
        started.set()
        release.wait(5.0)
        return "first"

    def quick(progress=None, should_cancel=None) -> str:
        release.set()
        return "second"

    state = AppState()
    state._start(Job(slow), on_finished=lambda _: None, on_failed=lambda *_: None)
    assert started.wait(5.0)
    state._start(Job(quick), on_finished=lambda _: None, on_failed=lambda *_: None)
    second = state._job

    qtbot.waitUntil(lambda: not state.busy, timeout=10_000)
    assert second is not None
    assert state._job is None


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


def test_unavailable_stages_say_what_would_open_them(qtbot) -> None:
    """A shut door has to name its key.

    The rail used to mark six of nine steps ``(not yet)``, which tells the user
    that most of the program is closed without telling them what closes it. "Open
    a log first" and "runs with the analysis" are different problems with
    different fixes, and the user can act on either.
    """
    from rotorid.gui.main_window import MainWindow

    window = MainWindow(AppState())
    qtbot.addWidget(window)

    design = window.rail.step(STAGES.index("Design"))
    assert design.state == "waiting"
    assert "log" in design.note.lower(), design.note
    assert "not yet" not in design.note

    load = window.rail.step(0)
    assert load.open
    assert load.note == load.blurb, "an open step describes itself rather than excusing itself"


def test_every_rail_step_says_what_it_is_for(qtbot) -> None:
    """Nine pieces of jargon are a menu; nine jargon terms with glosses are a sequence."""
    from rotorid.gui.main_window import MainWindow

    window = MainWindow(AppState())
    qtbot.addWidget(window)
    for row in range(window.rail.count()):
        step = window.rail.step(row)
        assert step is not None and step.blurb, step
        assert step.name in STAGES


def test_a_gated_stage_stays_clickable_backwards_but_not_forwards(qtbot, log: Path) -> None:
    """Going back is always allowed; going forward needs something to look at."""
    from rotorid.gui.main_window import MainWindow

    state = AppState()
    state.auto_analyse = False
    window = MainWindow(state)
    qtbot.addWidget(window)

    assert not window.next_button.isEnabled(), "nothing to go on to with no log"
    with qtbot.waitSignal(state.log_loaded, timeout=60_000):
        state.load_log(log)
    assert window.next_button.isEnabled(), "the noise screen only needs the log"
    # Doubled, because Qt reads a lone ampersand on a button as a mnemonic and
    # would draw "Health _N_oise". What the user sees is one ampersand.
    assert window.next_button.text() == "Next: Health && Noise"


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


# --------------------------------------------------------------------------- #
# The frequency axis
# --------------------------------------------------------------------------- #

#: Ranges a frequency plot in this tool actually shows, in log10 Hz, and the
#: widths one gets at on a laptop with the findings dock open.
_BANDS = (
    (-0.15, 2.6),  # 0.7 Hz - 400 Hz, a noise spectrum
    (0.0, 2.0),  # 1 Hz - 100 Hz, a Bode plot
    (-1.0, 3.5),  # 0.1 Hz - 3 kHz, zoomed out
    (0.6, 1.3),  # 4 Hz - 20 Hz, zoomed in
)
_WIDTHS = (1100.0, 750.0, 500.0, 380.0, 260.0)


def _axis(qtbot):
    from rotorid.gui.widgets.log_axis import LogAxis

    axis = LogAxis("bottom")
    axis.setLogMode(True)
    return axis


def test_frequency_labels_never_overlap(qtbot) -> None:
    """The bug this axis exists for.

    pyqtgraph returns every tick in log mode as one level, and its crowding rule
    exempts the first level on the assumption that a first level is sparse. Over
    three decades that draws twenty-odd labels into one axis width, and the scale
    comes out as a smear. Asserted in pixels, because "does it overlap" is a
    question about pixels.
    """
    from PySide6.QtGui import QFontMetricsF

    axis = _axis(qtbot)
    metrics = QFontMetricsF(axis.font())

    for lo, hi in _BANDS:
        for size in _WIDTHS:
            values = axis.logTickValues(lo, hi, size, [])[0][1]
            strings = axis.tickStrings(values, 1.0, None)
            centres = [(v - lo) / (hi - lo) * size for v in values]
            widths = [metrics.horizontalAdvance(s) for s in strings]
            for i in range(len(values) - 1):
                clear = centres[i + 1] - centres[i] - (widths[i] + widths[i + 1]) / 2
                assert clear >= 0.0, (
                    f"{strings[i]!r} and {strings[i + 1]!r} collide by {-clear:.0f}px "
                    f"over {10**lo:.3g}-{10**hi:.3g} Hz at {size:.0f}px"
                )


def test_a_narrow_axis_sheds_labels_rather_than_stacking_them(qtbot) -> None:
    """Fewer labels is the right answer to less room; the same labels smaller is not."""
    axis = _axis(qtbot)
    lo, hi = _BANDS[0]
    wide = len(axis.logTickValues(lo, hi, 1100.0, [])[0][1])
    narrow = len(axis.logTickValues(lo, hi, 260.0, [])[0][1])
    assert narrow < wide
    assert narrow >= 2, "an axis with one label on it is not a scale"


def test_the_minor_ticks_are_marks_rather_than_more_text(qtbot) -> None:
    """The decade structure stays visible without competing for space as labels."""
    axis = _axis(qtbot)
    levels = axis.logTickValues(*_BANDS[0], 500.0, [])
    assert len(levels) == 3, "labels that fit, labels if there is room, then bare marks"
    assert levels[2][1], "the minor ticks are still drawn"
    assert axis.style["maxTextLevel"] == 1, "so the third level can never be labelled"


def test_frequencies_are_written_the_way_they_are_spoken(qtbot) -> None:
    """``2·10¹`` is a number; ``20`` is a frequency somebody is about to type in."""
    axis = _axis(qtbot)
    seen: set[str] = set()
    for lo, hi in _BANDS:
        for size in _WIDTHS:
            for _, values in axis.logTickValues(lo, hi, size, []):
                seen.update(axis.tickStrings(values, 1.0, None))

    assert seen
    assert not any("·" in s or "e" in s or "^" in s for s in seen), sorted(seen)
    assert {"1", "10", "100"} <= seen
    assert all(s.replace(".", "").replace("k", "").isdigit() for s in seen), sorted(seen)
