"""The Design stage and its sandbox (milestone M7).

The sandbox is a product requirement, not polish: the user is supposed to learn
the trade by moving one control and watching four things move together. That
only works if the re-solve is genuinely live, so what is tested here is that
moving the slider actually changes the answer, that the change goes the
direction the control claims, and that it happens inside the interactive budget.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rotorid.gui.state import AppState
from tests.synthetic.generators import make_airframe, make_bundle, make_chain

pytest.importorskip("PySide6")


@pytest.fixture
def state(qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppState:
    """An analysed session, ready for the Design stage to draw."""
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
def stage(qtbot, state: AppState):
    from rotorid.gui.wizard.design import DesignStage

    widget = DesignStage(state)
    qtbot.addWidget(widget)
    widget.refresh()
    return widget


def test_the_stage_opens_on_the_recommendation(stage, state: AppState) -> None:
    recommended = next(iter(state.result.session.recommendations.values()))
    assert stage._live is not None
    assert stage._live.gains.kp == pytest.approx(recommended.gains.kp)


def test_moving_the_slider_changes_the_design(qtbot, stage) -> None:
    """A sandbox that does not move is a picture of a sandbox."""
    before = stage._live.gains.kp

    stage._slider.setValue(95)
    qtbot.waitUntil(lambda: stage._live.gains.kp != before, timeout=5_000)

    assert stage._live.gains.kp != before


def test_more_conservatism_buys_margin_and_costs_bandwidth(qtbot, stage) -> None:
    """The direction of the trade the control claims to make."""

    def solve_at(value: int):
        stage._slider.setValue(value)
        stage._resolve()
        return stage._live

    aggressive = solve_at(5)
    pm_low = aggressive.margins.phase_margin_deg
    drb_low = aggressive.margins.disturbance_rejection_bw_hz

    docile = solve_at(95)
    assert docile.margins.phase_margin_deg >= pm_low
    assert docile.margins.disturbance_rejection_bw_hz <= drb_low


def test_the_resolve_stays_inside_the_interactive_budget(stage) -> None:
    """Reported on screen, so a regression is visible rather than merely felt.

    Measured after one warm-up solve. The budget is a claim about *dragging* the
    slider -- the state the user spends their time in -- and the very first solve
    after the stage opens pays for imports and cache fills that no later one
    does. The screen reports the last solve either way, so a genuine regression
    still shows up in front of the user.
    """
    stage._slider.setValue(60)
    stage._resolve()

    stage._slider.setValue(70)
    stage._resolve()
    text = stage._timing.text()
    assert "re-solved in" in text
    assert "over the 300 ms" not in text, text


def test_the_binding_constraint_is_shown_in_plain_language(stage) -> None:
    text = stage._binding.text()
    assert text.startswith("What stops the gains going higher")
    assert "_" not in text, "a constraint identifier is not an explanation"


def test_every_gain_can_answer_why_it_is_that_number(stage) -> None:
    """Spec section 0.3: a value that cannot answer this is a bug."""
    from PySide6.QtWidgets import QPushButton

    buttons = [b for b in stage._gains_card.findChildren(QPushButton) if b.text() == "why?"]
    assert len(buttons) == 4, "P, I, D and FF each need their own trace"
    assert all(b.toolTip() for b in buttons)


def test_the_baseline_stays_on_the_plots(stage) -> None:
    """Exploring must never lose the reference point the user came in with."""
    names = {item.name() for item in stage._step.plot.getPlotItem().listDataItems() if item.name()}
    assert names == {"Current", "Recommended"}


def test_only_the_axes_that_were_analysed_get_a_button(qtbot, state: AppState) -> None:
    """An axis with no sweep in the log has nothing to design, and says so by absence."""
    from PySide6.QtWidgets import QPushButton

    from rotorid.gui.wizard.design import DesignStage

    with qtbot.waitSignal(state.analysis_finished, timeout=180_000):
        state.run_analysis(("roll", "pitch", "yaw"))

    widget = DesignStage(state)
    qtbot.addWidget(widget)
    widget.refresh()

    analysed = set(state.result.session.recommendations)
    labels = {
        b.text().lower()
        for b in widget.findChildren(QPushButton)
        if b.text().lower() in {"roll", "pitch", "yaw"}
    }
    assert labels == analysed
    assert widget._axis in analysed


def test_picking_an_unanalysed_axis_falls_back_rather_than_blanking(stage) -> None:
    """The screen must never end up showing nothing with no explanation."""
    stage._pick_axis("yaw")
    assert stage._live is not None
    assert stage._axis in stage.state.result.session.recommendations


def test_the_slider_stops_where_the_designer_stops(
    qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A control that moves and changes nothing is worse than one that will not move.

    A general flight log is designed at no less than its conservatism floor
    whatever the slider says. If the slider still travelled below it, the user
    would read the number under their thumb and believe it.
    """
    from rotorid.core.io import ardupilot
    from rotorid.core.logkind import capabilities
    from rotorid.gui.main_window import MainWindow
    from rotorid.gui.state import AppState
    from tests.synthetic.generators import make_general_flight_bundle

    path = tmp_path / "general.bin"
    path.write_bytes(b"")
    bundle = make_general_flight_bundle(make_airframe(), make_chain())
    monkeypatch.setattr(ardupilot, "read_ardupilot", lambda p, **kw: bundle)

    win = MainWindow(AppState())
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.state.log_loaded, timeout=60_000):
        win.state.load_log(path)
    with qtbot.waitSignal(win.state.analysis_finished, timeout=180_000):
        win.state.run_analysis(("roll",))

    stage = win.design_stage
    stage.refresh()
    floor = capabilities("general").conservatism_floor
    assert stage._slider.minimum() == round(100 * floor)
    assert not stage._floor.isHidden()

    stage._slider.setValue(0)
    assert stage._slider.value() == round(100 * floor), "the slider must not travel below it"
