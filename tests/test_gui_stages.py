"""Every stage, drawn from one real analysis (milestones M6/M7).

A per-stage smoke pass, plus the properties that are easy to lose while stages
are being built independently: that stepping through the whole rail draws
something on each screen without raising, that the observational stages are
honest about provenance, and that the rail never leaves the user on a screen
that is silently empty.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from rotorid.gui.state import STAGES, AppState
from tests.synthetic.generators import make_airframe, make_bundle, make_chain

pytest.importorskip("PySide6")


@pytest.fixture
def window(qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from rotorid.core.io import ardupilot
    from rotorid.gui.main_window import MainWindow

    path = tmp_path / "flight.bin"
    path.write_bytes(b"")
    bundle = make_bundle(make_airframe(), make_chain(), with_motor_noise=True)
    monkeypatch.setattr(ardupilot, "read_ardupilot", lambda p, **kw: bundle)

    win = MainWindow(AppState())
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.state.log_loaded, timeout=60_000):
        win.state.load_log(path)
    with qtbot.waitSignal(win.state.analysis_finished, timeout=180_000):
        win.state.run_analysis(("roll",))
    return win


def _labels(widget) -> str:
    return " | ".join(label.text() for label in widget.findChildren(QLabel) if label.text())


# --------------------------------------------------------------------------- #
# The rail
# --------------------------------------------------------------------------- #


def test_every_stage_in_the_rail_draws_without_raising(window) -> None:
    for row, name in enumerate(STAGES):
        window.rail.setCurrentRow(row)
        assert window.work.currentIndex() == row, name
        window.work.currentWidget().refresh()


def test_every_stage_is_reachable_once_the_analysis_has_run(window) -> None:
    assert all(window.state.stage_ready(name) for name in STAGES)
    assert all("not yet" not in window.rail.item(row).text() for row in range(len(STAGES)))


# --------------------------------------------------------------------------- #
# The observational stages
# --------------------------------------------------------------------------- #


def test_health_says_where_the_pre_filter_spectrum_came_from(window) -> None:
    """A reconstruction is blind where a notch is deep, and must say so."""
    stage = window.health_stage
    stage.refresh()
    text = _labels(stage)
    assert "peak(s)" in text
    assert any(word in text for word in ("reconstructed", "measured directly", "not available"))


def test_health_names_each_peak_and_what_to_do_about_it(window) -> None:
    stage = window.health_stage
    stage.refresh()
    if stage._peaks.rowCount() == 0:
        pytest.skip("this log has no peaks above the floor")
    kinds = {stage._peaks.item(r, 3).text() for r in range(stage._peaks.rowCount())}
    advice = {stage._peaks.item(r, 4).text() for r in range(stage._peaks.rowCount())}
    assert kinds
    assert all(advice), "a classification with no consequence teaches nothing"


def test_segment_shades_the_stretches_the_identification_used(window) -> None:
    import pyqtgraph as pg

    stage = window.segment_stage
    stage.refresh()
    items = stage._trace.plot.getPlotItem().items
    regions = [i for i in items if isinstance(i, pg.LinearRegionItem)]
    assert regions, "the excitation was not marked on the trace"
    assert stage._table.rowCount() == len(regions)


def test_identify_draws_the_measurement_the_chain_and_the_airframe(window) -> None:
    """The effective-plant / airframe distinction, made visible."""
    stage = window.identify_stage
    stage.refresh()
    names = {
        item.name() for item in stage._magnitude.plot.getPlotItem().listDataItems() if item.name()
    }
    assert names == {
        "Measured (filters included)",
        "Modelled filter chain",
        "Airframe, fitted",
    }


def test_identify_states_the_band_the_model_is_evidence_over(window) -> None:
    stage = window.identify_stage
    stage.refresh()
    text = stage._verdict.text()
    assert "Fitted over" in text
    assert "mean coherence" in text
    assert "Filters removed by" in text


def test_next_flight_tells_the_pilot_how_to_back_a_gain_off_in_the_air(window) -> None:
    """The most useful safety property a tuning flight can have."""
    stage = window.nextflight_stage
    stage.refresh()
    text = _labels(stage)
    assert "TUNE" in text
    assert "oscillate" in text
    assert "Flight 1:" in text


# --------------------------------------------------------------------------- #
# The shell
# --------------------------------------------------------------------------- #


def test_the_findings_dock_counts_what_is_blocking(window) -> None:
    window._refresh_findings()
    blocked = window.state.unresolved
    title = window.findings_dock.windowTitle()
    assert ("blocking" in title) == bool(blocked)


def test_a_plot_can_explain_itself(window) -> None:
    """Spec section 10.3: a plot nobody can interpret teaches nothing."""
    for card in (
        window.identify_stage._magnitude,
        window.identify_stage._coherence,
        window.filters_stage._spectrum,
        window.design_stage._step,
        window.segment_stage._trace,
    ):
        assert len(card._explanation) > 200, "an explanation has to actually explain"


# --------------------------------------------------------------------------- #
# The vibration gate
# --------------------------------------------------------------------------- #


@pytest.fixture
def shaking_window(qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The same window, from a log whose frame was shaking hard enough to matter."""
    from dataclasses import replace

    import numpy as np

    from rotorid.core.io import ardupilot
    from rotorid.core.io.base import canonical_signal
    from rotorid.gui.main_window import MainWindow

    path = tmp_path / "shaking.bin"
    path.write_bytes(b"")
    bundle = make_bundle(make_airframe(), make_chain(), with_motor_noise=True)
    t = bundle.signals["rate.roll.measured"].t
    signals = dict(bundle.signals)
    signals["imu.0.vibe.z"] = canonical_signal(
        "imu.0.vibe.z", t, np.full(t.shape, 45.0), source_msg="VIBE"
    )
    monkeypatch.setattr(
        ardupilot, "read_ardupilot", lambda p, **kw: replace(bundle, signals=signals)
    )

    win = MainWindow(AppState())
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.state.log_loaded, timeout=60_000):
        win.state.load_log(path)
    with qtbot.waitSignal(win.state.analysis_finished, timeout=180_000):
        win.state.run_analysis(("roll",))
    return win


def test_health_admits_when_vibration_was_never_logged(window) -> None:
    """Silence is not a clean bill of health, and the screen must not imply it is."""
    stage = window.health_stage
    stage.refresh()
    assert not stage._gate.isHidden()
    assert "cannot tell" in stage._gate.text()


def test_a_shaking_frame_is_called_out_above_the_spectrum(shaking_window) -> None:
    """The gate has to sit above the peak table, because it invalidates it."""
    stage = shaking_window.health_stage
    stage.refresh()
    assert not stage._gate.isHidden()
    assert "m/s^2" in stage._gate.text()
    assert "mechanical" in stage._gate.text().lower()

    layout = stage.layout()
    order = [layout.itemAt(i).widget() for i in range(layout.count())]
    assert order.index(stage._gate) < order.index(stage._peaks)


# --------------------------------------------------------------------------- #
# Starting with nothing
# --------------------------------------------------------------------------- #


@pytest.fixture
def empty_window(qtbot):
    """The window as it opens when nobody named a log on the command line."""
    from rotorid.gui.main_window import MainWindow

    win = MainWindow(AppState())
    qtbot.addWidget(win)
    return win


def test_the_window_opens_without_a_log_and_offers_to_find_one(empty_window) -> None:
    """The reason `rotorid` on its own is allowed to open a window at all."""
    stage = empty_window.load_stage
    assert empty_window.work.currentIndex() == 0
    assert stage._choose.isEnabled()
    assert stage._choose.isDefault(), "the one thing to do here should be the default action"
    assert not stage._empty.isHidden(), "an empty page with a button reads as a broken page"
    assert stage._signals.isHidden(), "there are no signals to list yet"


def test_a_log_can_be_dropped_on_the_window_not_only_on_the_load_page(
    empty_window, qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PySide6.QtCore import QMimeData, QPoint, QUrl
    from PySide6.QtGui import QDropEvent

    from rotorid.core.io import ardupilot

    path = tmp_path / "dropped.bin"
    path.write_bytes(b"")
    bundle = make_bundle(make_airframe(), make_chain())
    monkeypatch.setattr(ardupilot, "read_ardupilot", lambda p, **kw: bundle)

    empty_window.rail.setCurrentRow(len(STAGES) - 1)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    event = QDropEvent(
        QPoint(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    # The handler directly rather than through Qt's drag machinery: what is being
    # tested is that the window has one and that it does the right thing, not that
    # Qt delivers events.
    with qtbot.waitSignal(empty_window.state.log_loaded, timeout=60_000):
        empty_window.dropEvent(event)

    assert empty_window.state.bundle is bundle
    # And it brings the user back to the page that describes what was just opened.
    assert empty_window.work.currentIndex() == 0


def test_the_kind_of_flight_is_asked_before_the_file_is_chosen(empty_window) -> None:
    """The one question the file cannot answer for itself (spec 5.2).

    On the first screen, above the picker, because it decides what the load is
    for. Discovering it after a refusal is discovering it too late.
    """
    stage = empty_window.load_stage
    assert set(stage._kind_buttons) == {None, "general", "tuning"}
    assert stage._kind_buttons[None].isChecked(), "detection is the default"
    assert empty_window.state.declared_kind is None


def test_choosing_a_kind_is_recorded_and_reaches_the_reader(
    empty_window, qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The declaration has to arrive at the read, not be applied afterwards.

    Which segments are searched for depends on it, so a bundle read one way and
    relabelled the other is a lie every later stage builds on.
    """
    from rotorid.core.io import ardupilot

    path = tmp_path / "flight.bin"
    path.write_bytes(b"")
    seen: list[object] = []

    def reader(p, progress=None, *, kind=None):
        seen.append(kind)
        return dataclasses.replace(make_bundle(make_airframe(), make_chain()), declared_kind=kind)

    monkeypatch.setattr(ardupilot, "read_ardupilot", reader)

    empty_window.load_stage._kind_buttons["general"].setChecked(True)
    assert empty_window.state.declared_kind == "general"

    with qtbot.waitSignal(empty_window.state.log_loaded, timeout=60_000):
        empty_window.state.load_log(path)
    assert seen == ["general"]
    assert empty_window.state.bundle.kind == "general"


def test_changing_the_kind_re_reads_the_open_log(
    empty_window, qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reinterpreting in place would leave the label and the segments disagreeing."""
    from rotorid.core.io import ardupilot

    path = tmp_path / "flight.bin"
    path.write_bytes(b"")
    reads: list[object] = []

    def reader(p, progress=None, *, kind=None):
        reads.append(kind)
        return dataclasses.replace(make_bundle(make_airframe(), make_chain()), declared_kind=kind)

    monkeypatch.setattr(ardupilot, "read_ardupilot", reader)
    with qtbot.waitSignal(empty_window.state.log_loaded, timeout=60_000):
        empty_window.state.load_log(path)

    with qtbot.waitSignal(empty_window.state.log_loaded, timeout=60_000):
        empty_window.state.declare_kind("tuning")

    assert reads == [None, "tuning"]


def test_the_load_page_says_what_the_declaration_costs(window) -> None:
    """Stated where the choice was made, not discovered on the Review screen."""
    text = window.load_stage._verdict.text()
    assert "tuning flight" in text.lower()
    assert not window.load_stage._verdict.isHidden()


# --------------------------------------------------------------------------- #
# Validate (M10)
# --------------------------------------------------------------------------- #


@pytest.fixture
def closed_loop_window(qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A window on a closed-loop flight, which is what validation needs.

    The sweep fixture logs no rate setpoint, and everything the Validate stage
    reads is a comparison of what was asked for against what happened.
    """
    from rotorid.core.io import ardupilot
    from rotorid.gui.main_window import MainWindow
    from tests.synthetic.closed_loop import make_closed_loop_bundle

    path = tmp_path / "before.bin"
    path.write_bytes(b"")
    bundle = make_closed_loop_bundle(path="before.bin")
    monkeypatch.setattr(ardupilot, "read_ardupilot", lambda p, **kw: bundle)

    win = MainWindow(AppState())
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.state.log_loaded, timeout=60_000):
        win.state.load_log(path)
    return win


def test_validate_is_reachable_before_anything_has_been_analysed(closed_loop_window) -> None:
    """A before/after on tracking error is worth having even with no model.

    Gating this on a successful identification would shut the door on exactly the
    user whose logs cannot be identified -- who is the user with the most to gain
    from being told, in numbers, whether the change helped.
    """
    assert closed_loop_window.state.stage_ready("Validate")
    row = STAGES.index("Validate")
    closed_loop_window.rail.setCurrentRow(row)
    assert closed_loop_window.work.currentIndex() == row


def test_validate_says_it_is_not_a_validation_until_the_analysis_has_run(
    closed_loop_window, qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The distinction the whole screen exists to protect."""
    stage = closed_loop_window.validate_stage
    after = tmp_path / "after.bin"
    after.write_bytes(b"")

    with qtbot.waitSignal(closed_loop_window.state.comparison_finished, timeout=180_000):
        closed_loop_window.state.load_after_log(after)
    stage.refresh()

    assert "not a validation" in stage._scope.text()
    assert closed_loop_window.state.comparison is not None


def test_with_an_analysis_the_prediction_is_on_the_screen(
    closed_loop_window, qtbot, tmp_path: Path
) -> None:
    stage = closed_loop_window.validate_stage
    with qtbot.waitSignal(closed_loop_window.state.analysis_finished, timeout=180_000):
        closed_loop_window.state.run_analysis(("roll",))

    after = tmp_path / "after.bin"
    after.write_bytes(b"")
    with qtbot.waitSignal(closed_loop_window.state.comparison_finished, timeout=180_000):
        closed_loop_window.state.load_after_log(after)
    stage.refresh()

    assert "This is a validation" in stage._scope.text()
    assert stage._table.topLevelItemCount() >= 1
    assert stage._table.topLevelItem(0).text(0) == "roll"


def test_loading_a_second_log_does_not_throw_the_first_one_away(
    closed_loop_window, qtbot, tmp_path: Path
) -> None:
    """The whole point of the screen is that both logs are on it at once."""
    before = closed_loop_window.state.bundle
    after = tmp_path / "after.bin"
    after.write_bytes(b"")
    with qtbot.waitSignal(closed_loop_window.state.comparison_finished, timeout=180_000):
        closed_loop_window.state.load_after_log(after)

    assert closed_loop_window.state.bundle is before
    assert closed_loop_window.state.after is not None
