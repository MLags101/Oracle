"""The Filters stage and its sandbox (milestone M7).

The stage's whole claim is that an override is *measured*, not merely accepted:
a hand-built chain goes through the same solve as the designed one, so the phase
cost, the D-term noise and the margins beside it describe that chain. The tests
below are mostly about that claim, plus the honesty requirements -- a
reconstructed pre-filter spectrum has to say it is reconstructed, and a stage
with nothing to recommend has to say so rather than sit blank.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rotorid.gui.state import AppState
from tests.synthetic.generators import make_airframe, make_bundle, make_chain

pytest.importorskip("PySide6")


@pytest.fixture
def state(qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppState:
    """A session analysed from a log with real motor noise in it."""
    from rotorid.core.io import ardupilot

    path = tmp_path / "flight.bin"
    path.write_bytes(b"")
    bundle = make_bundle(make_airframe(), make_chain(), with_motor_noise=True)
    monkeypatch.setattr(ardupilot, "read_ardupilot", lambda p, **kw: bundle)

    app_state = AppState()
    with qtbot.waitSignal(app_state.log_loaded, timeout=60_000):
        app_state.load_log(path)
    with qtbot.waitSignal(app_state.analysis_finished, timeout=180_000):
        app_state.run_analysis(("roll",))
    return app_state


@pytest.fixture
def stage(qtbot, state: AppState):
    from rotorid.gui.wizard.filters import FiltersStage

    widget = FiltersStage(state)
    qtbot.addWidget(widget)
    widget.refresh()
    return widget


def test_the_controls_open_on_the_recommended_chain(stage) -> None:
    chain = stage._live.filters.chain
    if chain.gyro_lpf_hz:
        assert stage._gyro_lpf.currentData() == pytest.approx(chain.gyro_lpf_hz)
    if chain.notches:
        notch = chain.notches[0]
        assert stage._notch_bw.value() == round(notch.bandwidth_hz)
        assert stage._notch_att.value() == round(notch.attenuation_db)
        for n, box in stage._harmonics.items():
            assert box.isChecked() == (n in notch.harmonics)


def test_an_override_is_measured_rather_than_accepted(stage) -> None:
    """A hand-built chain gets the same yardsticks as the designed one."""
    before = stage._live
    if not before.filters.chain.notches:
        pytest.skip("this fixture recommends no notch")

    stage._notch_bw.setValue(min(before.filters.chain.notches[0].bandwidth_hz * 2, 199))
    stage._resolve()
    after = stage._live

    assert after is not before
    assert after.filters.chain.notches[0].bandwidth_hz == pytest.approx(stage._notch_bw.value())
    assert after.filters.phase_cost_deg != before.filters.phase_cost_deg, (
        "a wider notch costs more phase; if this number did not move, "
        "the override was displayed but not solved"
    )


def test_a_wider_notch_costs_phase(stage) -> None:
    """The lesson the sandbox exists to teach, asserted as a property.

    Compared at one fixed frequency rather than at each design's own crossover.
    Phase lag is a function of frequency, and a wider notch *moves* the
    crossover, so the two headline numbers on screen are not measured at the same
    place and cannot be subtracted from each other -- which is exactly why the
    phase budget is drawn at a stated frequency and labelled with it.
    """
    import numpy as np

    if not stage._live.filters.chain.notches:
        pytest.skip("this fixture recommends no notch")

    probe = np.array([stage._live.margins.crossover_hz])

    def chain_at(bandwidth: int):
        stage._notch_bw.setValue(bandwidth)
        stage._resolve()
        return stage._live.filters.chain

    narrow = float(chain_at(20).phase_deg(probe)[0])
    wide = float(chain_at(80).phase_deg(probe)[0])
    assert abs(wide) > abs(narrow)


def test_a_worse_override_is_allowed_and_shown_as_worse(stage) -> None:
    """It is a sandbox, not a set of presets. It must be able to disappoint."""
    recommended_drb = stage._live.margins.disturbance_rejection_bw_hz

    index = stage._gyro_lpf.findData(20.0)
    if index < 0:
        pytest.skip("20 Hz is not on the ladder")
    stage._gyro_lpf.setCurrentIndex(index)
    stage._resolve()

    assert stage._live.filters.chain.gyro_lpf_hz == pytest.approx(20.0)
    assert stage._live.margins.disturbance_rejection_bw_hz <= recommended_drb


def test_rebuilding_the_flown_chain_asks_for_no_parameter_change(stage, state) -> None:
    """ "The user recreated what they have" and "nothing recommended" are the same."""
    flown = state.result.analyses[stage._axis].chain
    stage._live = stage._live  # nothing to change; use the flown chain directly
    from rotorid.core.design.filters import describe_chain

    recommendation = describe_chain(
        flown,
        flown,
        axis="roll",
        stack="ardupilot",
        noise=None,
        op=None,
        crossover_hz=5.0,
    )
    assert recommendation.params == {}
    assert recommendation.chain is recommendation.baseline_chain


def test_the_spectrum_says_when_the_pre_filter_trace_is_reconstructed(stage) -> None:
    """A reconstruction is blind exactly where a working notch is deepest."""
    names = {
        item.name() for item in stage._spectrum.plot.getPlotItem().listDataItems() if item.name()
    }
    assert names, "nothing was plotted"
    if stage._pre_filter_source() == "reconstructed":
        assert any("reconstructed" in name for name in names)


def test_nothing_to_change_is_stated_rather_than_left_blank(stage) -> None:
    if stage._live.filters.params:
        assert stage._diff.rowCount() == len(stage._live.filters.params)
    else:
        assert stage._diff.rowCount() == 1
        assert "already flying" in stage._diff.item(0, 0).text()


def test_the_phase_cost_is_stated_against_the_margin_it_comes_out_of(stage) -> None:
    text = stage._verdict.text()
    assert "deg of phase at the" in text
    assert "D-term noise" in text


def test_the_filter_values_can_answer_why(stage) -> None:
    from PySide6.QtWidgets import QPushButton

    labels = {b.text() for b in stage.findChildren(QPushButton) if b.text().startswith("why ")}
    assert "why gyro low-pass?" in labels
