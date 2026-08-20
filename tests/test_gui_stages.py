"""Every stage, drawn from one real analysis (milestones M6/M7).

A per-stage smoke pass, plus the properties that are easy to lose while stages
are being built independently: that stepping through the whole rail draws
something on each screen without raising, that the observational stages are
honest about provenance, and that the rail never leaves the user on a screen
that is silently empty.
"""

from __future__ import annotations

from pathlib import Path

import pytest
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
