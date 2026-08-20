"""The M1 walking skeleton, end to end.

A synthetic log goes in and a traceable recommendation comes out. These tests
exist to catch the failures that only appear when the pieces are wired together:
a segment boundary off by a window, a parameter snapshot that rebuilds a
different chain than the one that filtered the data, an axis mix-up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rotorid.config import load_config
from rotorid.core.design.recommend import analyze_axis, identify_axis
from rotorid.core.export.report import write_report
from rotorid.core.preprocess.params import chain_from_bundle, gains_from_bundle
from rotorid.core.preprocess.segment import propose_segments
from tests.synthetic.generators import make_airframe, make_bundle, make_chain

CONFIG = load_config()


def _bundle(**kw):
    return make_bundle(kw.pop("airframe", make_airframe()), kw.pop("chain", make_chain()), **kw)


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #


def test_systemid_segment_is_found_and_attributed_to_the_right_axis() -> None:
    bundle = _bundle(axis="pitch")
    segments = propose_segments(bundle)

    assert len(segments) == 1
    assert segments[0].axis == "pitch"
    assert segments[0].kind == "systemid_chirp"
    assert segments[0].confidence == 1.0
    assert segments[0].injection_point == "rate"
    assert segments[0].duration_s > 80.0


def test_segment_bounds_come_from_the_injected_signal_not_the_parameters() -> None:
    """A sweep cut short by a mode change must produce a shorter segment.

    ``SID_T_REC`` records what was asked for. The chirp records what happened.
    """
    long_sweep = propose_segments(_bundle(duration_s=90.0))[0]
    short_sweep = propose_segments(_bundle(duration_s=40.0))[0]

    assert short_sweep.duration_s < long_sweep.duration_s
    # The amplitude fade means the first and last seconds are below the detection
    # threshold, so the segment is a little shorter than the record. That is the
    # right answer: those seconds carry almost no excitation.
    assert 0.8 * 40.0 < short_sweep.duration_s <= 40.0


# --------------------------------------------------------------------------- #
# Parameter round trip
# --------------------------------------------------------------------------- #


def test_chain_rebuilt_from_parameters_matches_the_chain_that_filtered_the_data() -> None:
    """The load-bearing assumption of the whole deconvolution.

    If the chain reconstructed from the parameter snapshot differs at all from the
    one the samples actually went through, the airframe is wrong by that
    difference -- and nothing downstream can detect it.
    """
    chain = make_chain()
    bundle = _bundle(chain=chain)
    rebuilt = chain_from_bundle(bundle, "roll")

    f = np.geomspace(0.5, 500.0, 200)
    assert np.allclose(rebuilt.sensor_response(f), chain.sensor_response(f), rtol=1e-12)
    assert rebuilt.dterm_lpf_hz == chain.dterm_lpf_hz
    assert rebuilt.loop_rate_hz == chain.loop_rate_hz


def test_current_gains_are_read_back_from_the_snapshot() -> None:
    bundle = _bundle(gains=(0.14, 0.14, 0.004))
    gains = gains_from_bundle(bundle, "roll")
    assert (gains.kp, gains.ki, gains.kd) == (0.14, 0.14, 0.004)


# --------------------------------------------------------------------------- #
# Identification through the full pipeline
# --------------------------------------------------------------------------- #


def test_pipeline_recovers_the_ground_truth_airframe() -> None:
    truth = make_airframe()
    analysis = identify_axis(_bundle(airframe=truth), "roll", CONFIG)
    model = analysis.airframe

    assert model.params["K"] == pytest.approx(truth.params["K"], rel=0.10)
    assert model.params["wn"] == pytest.approx(truth.params["wn"], rel=0.10)
    assert model.params["tau"] == pytest.approx(truth.params["tau"], rel=0.05)
    assert model.filter_deconvolution == "modeled"


def test_pipeline_uses_the_injected_chirp_when_it_is_available() -> None:
    analysis = identify_axis(_bundle(), "roll", CONFIG)
    assert analysis.effective.source == "injected_chirp"
    assert analysis.effective.filters_included is True


def test_pipeline_refuses_an_axis_that_was_never_excited() -> None:
    bundle = _bundle(axis="roll")
    with pytest.raises(ValueError, match="no usable excitation found on yaw"):
        identify_axis(bundle, "yaw", CONFIG)


def test_recommendation_is_complete_and_self_consistent() -> None:
    rec = analyze_axis(_bundle(), "roll", CONFIG)

    assert rec.axis == "roll"
    assert rec.gains.kp > 0.0
    assert rec.baseline_gains.kp == 0.135
    assert rec.margins.phase_margin_deg >= CONFIG.float_("margins", "pm_floor_deg")
    assert rec.margins.gain_margin_db >= CONFIG.float_("margins", "gm_min_db") - 0.1
    assert rec.binding_constraint
    assert rec.confidence in ("high", "medium", "low")
    assert rec.latency.at_hz == pytest.approx(rec.margins.crossover_hz)
    assert "divided out" in rec.rationale


def test_recommendation_beats_the_stock_tune_it_started_from() -> None:
    """The point of the exercise: more disturbance rejection at legal margins.

    Stock ArduPilot gains are deliberately conservative, so a design that cannot
    improve on them against the same airframe and the same filters has either a
    broken optimizer or a broken plant model.
    """
    from rotorid.core.analysis.margins import broken_loop, compute_margins, design_grid
    from rotorid.core.design.controller import controller_for

    bundle = _bundle()
    analysis = identify_axis(bundle, "roll", CONFIG)
    rec = analyze_axis(bundle, "roll", CONFIG)

    grid = design_grid(0.1, 200.0, 900)
    stock = compute_margins(
        grid,
        broken_loop(
            grid,
            controller_for("ardupilot", rec.baseline_gains, analysis.chain),
            analysis.airframe,
            delay=analysis.delay,
        ),
    )
    assert rec.margins.disturbance_rejection_bw_hz > stock.disturbance_rejection_bw_hz, (
        "the recommendation should reject disturbances better than the stock tune"
    )


def test_conservatism_reaches_the_recommendation() -> None:
    bundle = _bundle()
    fast = analyze_axis(bundle, "roll", CONFIG, conservatism=0.0)
    slow = analyze_axis(bundle, "roll", CONFIG, conservatism=1.0)
    assert slow.margins.crossover_hz < fast.margins.crossover_hz
    assert slow.conservatism == 1.0


def test_filters_and_gains_come_out_as_one_package() -> None:
    """A recommendation is never gains alone, and never filters alone.

    On a real vehicle the filter configuration is usually what limits the
    achievable bandwidth, so a recommendation that changed gains against a chain
    it had not examined would be answering the easier half of the question.
    """
    rec = analyze_axis(_bundle(), "roll", CONFIG)

    assert rec.filters.baseline_chain.gyro_lpf_hz == 60.0, "the flown chain is carried through"
    assert rec.filters.rationale
    assert np.isfinite(rec.dterm_noise_rms_pct), (
        "the D-term noise the design produces must be a measured number, not a hope"
    )
    if rec.filters.chain is not rec.filters.baseline_chain:
        assert rec.filters.params, "a proposed change must come with parameters to write"


def test_a_notch_is_not_removed_on_the_strength_of_the_quiet_it_produced() -> None:
    """The most dangerous filter recommendation the tool could make.

    A working notch hides its own peak. Reconstructing the pre-filter spectrum
    recovers it only down to the deconvolution floor -- past that, the log simply
    does not say what was there. Concluding "no peak, so no notch needed" from
    that silence would hand the user a vehicle that shakes itself apart.
    """
    chain = make_chain(notch_freq_hz=90.0, notch_att_db=40.0, harmonics=(1, 2))
    rec = analyze_axis(_bundle(chain=chain), "roll", CONFIG)

    assert rec.filters.chain.notches, "the flown notch must survive"
    assert "INS_HNTCH_ENABLE" not in rec.filters.params, (
        "keeping a notch means writing nothing, not rewriting the same values"
    )
    assert "kept as flown" in rec.filters.rationale


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def test_report_is_written_and_self_contained(tmp_path: Path) -> None:
    bundle = _bundle()
    rec = analyze_axis(bundle, "roll", CONFIG)
    out = write_report(
        tmp_path / "report.html",
        bundle,
        {"roll": rec},
        config_hash="abcd1234",
        tool_version="test",
    )
    text = out.read_text(encoding="utf-8")

    assert text.startswith("<!doctype html>")
    assert "<script" not in text, "the report must not depend on scripting"
    assert "http://" not in text and "https://" not in text, "no external assets"
    assert "Back up your" in text, "the safety block is mandatory"
    assert rec.binding_constraint in text
    assert "abcd1234" in text, "the config hash pins which numbers produced this"


def test_report_shows_the_phase_budget() -> None:
    bundle = _bundle()
    rec = analyze_axis(bundle, "roll", CONFIG)
    from rotorid.core.export.report import _budget_figure

    svg = _budget_figure(rec)
    assert "<svg" in svg
    assert "airframe delay" in svg


def test_a_continuous_sweep_is_found_even_though_its_envelope_is_flat() -> None:
    """The fallback detector has to see the best excitation, not only the worst.

    A deliberate slow-to-fast sweep -- the thing worth identifying from -- has a
    nearly constant envelope, so a rule that looks for energy several times an
    axis's own median finds bursts of stick input and misses the sweep entirely.
    """
    from rotorid.core.preprocess.segment import propose_segments
    from tests.synthetic.generators import make_airframe, make_bundle, make_chain

    bundle = make_bundle(make_airframe(), make_chain(), stack="px4", path="flight.ulg")
    assert "excite.roll" not in bundle.signals, "this fixture has no injected-signal message"

    segments = [s for s in propose_segments(bundle) if s.axis == "roll"]
    assert segments, "the sweep was not found"
    assert segments[0].kind == "pilot_input"
    assert segments[0].confidence < 1.0, "found by energy, and it has to say so"
    assert segments[0].duration_s > 30.0


def test_a_still_flight_produces_no_segments_at_all() -> None:
    """The other half of the same rule: absence of excitation is a real answer."""
    import numpy as np

    from rotorid.core.io.base import canonical_signal
    from rotorid.core.preprocess.segment import propose_segments
    from tests.synthetic.generators import make_airframe, make_bundle, make_chain

    bundle = make_bundle(make_airframe(), make_chain(), stack="px4", path="flight.ulg")
    still = {
        key: (
            canonical_signal(key, sig.t, np.zeros_like(sig.y), source_msg=sig.source_msg)
            if key.endswith(".output")
            else sig
        )
        for key, sig in bundle.signals.items()
    }
    from dataclasses import replace

    assert propose_segments(replace(bundle, signals=still)) == ()
