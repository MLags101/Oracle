"""End-to-end identification against synthetic ground truth.

The acceptance criterion for milestone M1 lives here: a synthetic chirp, seen
through a real filter chain, must recover ``K, wn, zeta`` within 10% and ``tau``
within 5%. The double-counting guard lives here too -- it is the test that would
have caught the central error of spec revision 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotorid.core.analysis.spectra import choose_nperseg, combine, estimate_frf, log_smooth
from rotorid.core.analysis.sysid import deconvolve, fit_airframe
from rotorid.core.types import EffectivePlant, FrequencyResponse
from tests.synthetic.generators import chirp, make_airframe, make_chain, simulate_effective

SIM_RATE_HZ = 4000.0
LOG_RATE_HZ = 400.0
DECIMATION = int(SIM_RATE_HZ / LOG_RATE_HZ)
F_START, F_STOP = 0.2, 20.0
COHERENCE_THRESHOLD = 0.6
DECONV_FLOOR_DB = -20.0

WN_BOUNDS_HZ = (0.5, 40.0)
ZETA_BOUNDS = (0.05, 2.0)
TAU_BOUNDS_MS = (5.0, 80.0)


def _sweep(airframe, chain, *, duration_s: float = 90.0, noise=None, seed: int = 0):
    """Fly a sweep and return what a log would hold: command and post-filter gyro.

    Both are subsampled to the loop rate, because that is the rate ``RATE``
    messages are written at -- the analysis never sees the 4 kHz gyro stream.
    """
    t, u = chirp(
        sample_rate_hz=SIM_RATE_HZ,
        duration_s=duration_s,
        f_start_hz=F_START,
        f_stop_hz=F_STOP,
        amplitude=0.1,
        fade_s=4.0,
    )
    n = None if noise is None else noise(t, seed)
    y = simulate_effective(u, SIM_RATE_HZ, airframe, chain, noise=n)
    return t[::DECIMATION], u[::DECIMATION], y[::DECIMATION]


def _effective_plant(u, y, axis="roll") -> EffectivePlant:
    nperseg = choose_nperseg(u.size, LOG_RATE_HZ, f_lowest_hz=F_START, min_averages=5)
    est = estimate_frf(
        u,
        y,
        LOG_RATE_HZ,
        nperseg=nperseg,
        input_signal="rate.roll.output",
        output_signal="rate.roll.measured",
    )
    frf = est.to_frf(coherence_threshold=COHERENCE_THRESHOLD, band_hz=(F_START, F_STOP))
    return EffectivePlant(axis=axis, frf=frf, filters_included=True, source="mixer_cmd")


def _identify(airframe, chain, **kw):
    _, u, y = _sweep(airframe, chain, **kw)
    plant = _effective_plant(u, y)
    g_air = deconvolve(plant, chain, floor_db=DECONV_FLOOR_DB)
    return fit_airframe(
        g_air,
        wn_bounds_hz=WN_BOUNDS_HZ,
        zeta_bounds=ZETA_BOUNDS,
        tau_bounds_ms=TAU_BOUNDS_MS,
    )


# --------------------------------------------------------------------------- #
# The M1 acceptance criterion
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("K", "wn_hz", "zeta", "tau_ms"),
    [
        (12.0, 2.5, 1.0, 18.0),  # 5-inch quad
        (6.0, 1.4, 1.2, 30.0),  # heavier, slower, more lag
        (20.0, 4.0, 0.8, 10.0),  # small and stiff
    ],
)
def test_synthetic_chirp_recovers_ground_truth(
    K: float, wn_hz: float, zeta: float, tau_ms: float
) -> None:
    truth = make_airframe(K=K, wn_hz=wn_hz, zeta=zeta, tau_ms=tau_ms)
    chain = make_chain()
    model = _identify(truth, chain)

    assert model.params["K"] == pytest.approx(K, rel=0.10)
    assert model.params["wn"] == pytest.approx(2.0 * np.pi * wn_hz, rel=0.10)
    assert model.params["zeta"] == pytest.approx(zeta, rel=0.10)
    assert model.params["tau"] == pytest.approx(tau_ms / 1000.0, rel=0.05)
    assert model.filter_deconvolution == "modeled"


def test_fit_quality_is_reported_and_good_on_clean_data() -> None:
    model = _identify(make_airframe(), make_chain())
    assert model.fit_rms_db < 1.0
    assert model.fit_rms_deg < 5.0


def test_identification_survives_sensor_noise() -> None:
    """Motor tones enter pre-filter, exactly where they do on a real vehicle."""
    from tests.synthetic.generators import motor_noise

    truth = make_airframe()
    chain = make_chain()
    model = _identify(
        truth,
        chain,
        noise=lambda t, seed: motor_noise(t, broadband_rms=0.05, seed=seed),
    )
    assert model.params["wn"] == pytest.approx(truth.params["wn"], rel=0.10)
    assert model.params["tau"] == pytest.approx(truth.params["tau"], rel=0.15)


# --------------------------------------------------------------------------- #
# The double-counting guard
# --------------------------------------------------------------------------- #


def test_double_counting_guard() -> None:
    """Skipping the deconvolution must visibly corrupt the identified delay.

    The log's gyro is post-filter, so a fit that treats it as the bare airframe
    absorbs the whole filter chain's lag into ``tau``. If this test ever passes
    with the two routes agreeing, the filter chain has stopped being modeled and
    every downstream margin is wrong.
    """
    truth = make_airframe()
    chain = make_chain()
    _, u, y = _sweep(truth, chain)
    plant = _effective_plant(u, y)

    correct = fit_airframe(
        deconvolve(plant, chain, floor_db=DECONV_FLOOR_DB),
        wn_bounds_hz=WN_BOUNDS_HZ,
        zeta_bounds=ZETA_BOUNDS,
        tau_bounds_ms=TAU_BOUNDS_MS,
    )
    naive = fit_airframe(
        deconvolve(plant, None, floor_db=DECONV_FLOOR_DB),
        wn_bounds_hz=WN_BOUNDS_HZ,
        zeta_bounds=ZETA_BOUNDS,
        tau_bounds_ms=TAU_BOUNDS_MS,
    )

    assert correct.params["tau"] == pytest.approx(truth.params["tau"], rel=0.05)
    assert naive.params["tau"] > correct.params["tau"] * 1.3, (
        "the un-deconvolved fit should absorb the filter chain's lag into tau"
    )
    assert naive.filter_deconvolution == "none"


def test_deconvolution_refuses_to_filter_a_prefilter_measurement() -> None:
    """A raw-gyro measurement has no chain in it; dividing one out is a sign error."""
    truth = make_airframe()
    chain = make_chain()
    _, u, y = _sweep(truth, chain)
    plant = _effective_plant(u, y)
    raw = EffectivePlant(axis="roll", frf=plant.frf, filters_included=False, source="mixer_cmd")
    with pytest.raises(ValueError, match="count the chain twice"):
        deconvolve(raw, chain, floor_db=DECONV_FLOOR_DB)


def test_deconvolution_drops_bins_inside_deep_notches() -> None:
    """Dividing by a 40 dB notch turns noise into a fictitious resonance."""
    truth = make_airframe()
    chain = make_chain(notch_freq_hz=10.0, notch_bw_hz=5.0, harmonics=(1,))
    _, u, y = _sweep(truth, chain)
    plant = _effective_plant(u, y)
    g_air = deconvolve(plant, chain, floor_db=DECONV_FLOOR_DB)

    floor = 10.0 ** (DECONV_FLOOR_DB / 20.0)
    attenuated = np.abs(chain.sensor_response(g_air.f_hz)) < floor
    assert attenuated.any(), "fixture should put a deep notch inside the swept band"
    assert g_air.n_bins_lost_to_notches > 0
    assert not g_air.valid_mask[attenuated].any(), "no bin below the floor may survive"

    # The skirts stay usable: the notch is only ill-conditioned where it is deep.
    skirt = (np.abs(g_air.f_hz - 10.0) < 2.0) & ~attenuated
    assert g_air.valid_mask[skirt].any()


# --------------------------------------------------------------------------- #
# Spectral machinery
# --------------------------------------------------------------------------- #


def test_nperseg_prefers_resolution_over_averaging() -> None:
    """A short record still resolves the low end, at the cost of fewer averages."""
    n = int(20.0 * LOG_RATE_HZ)
    nperseg = choose_nperseg(n, LOG_RATE_HZ, f_lowest_hz=0.5, min_averages=5)
    assert nperseg >= 2.0 * LOG_RATE_HZ / 0.5
    assert nperseg <= n


def test_nperseg_refuses_an_impossible_record() -> None:
    with pytest.raises(ValueError, match="cannot resolve"):
        choose_nperseg(400, LOG_RATE_HZ, f_lowest_hz=0.2, min_averages=5)


def test_combining_sweeps_sharpens_the_coherence_gate() -> None:
    """Averaging repeated sweeps is ArduPilot's own methodology; check what it buys.

    Not "more valid bins" -- the opposite, and that is the point. Coherence
    computed from few averages is biased *upward*, so a single sweep declares
    bins valid out where nothing was excited. Merging drives that bias down, so
    coherence rises inside the swept band and falls outside it.
    """
    from tests.synthetic.generators import motor_noise

    truth, chain = make_airframe(), make_chain()
    estimates = []
    for seed in (1, 2, 3):
        _, u, y = _sweep(truth, chain, noise=lambda t, s, seed=seed: motor_noise(t, seed=seed))
        nperseg = choose_nperseg(u.size, LOG_RATE_HZ, f_lowest_hz=F_START, min_averages=5)
        estimates.append(
            estimate_frf(u, y, LOG_RATE_HZ, nperseg=nperseg, input_signal="u", output_signal="y")
        )

    single = estimates[0]
    merged = combine(estimates)
    assert merged.n_segments == sum(e.n_segments for e in estimates)

    f = merged.f_hz
    inside = (f >= 2.0 * F_START) & (f <= 0.8 * F_STOP)
    outside = f > 1.5 * F_STOP
    assert merged.coherence[inside].mean() >= single.coherence[inside].mean() - 0.01
    assert merged.coherence[outside].mean() < single.coherence[outside].mean()


def test_combine_rejects_mismatched_grids() -> None:
    truth, chain = make_airframe(), make_chain()
    _, u, y = _sweep(truth, chain, duration_s=60.0)
    a = estimate_frf(u, y, LOG_RATE_HZ, nperseg=2048, input_signal="u", output_signal="y")
    b = estimate_frf(u, y, LOG_RATE_HZ, nperseg=4096, input_signal="u", output_signal="y")
    with pytest.raises(ValueError, match="different frequency grids"):
        combine([a, b])


def test_log_smoothing_thins_the_high_end_without_inventing_bins() -> None:
    """Re-binning must reduce the grid, not fabricate resolution it never had.

    At the very bottom the FFT grid is sparser than the requested log grid, so
    those bins pass through one-to-one and stay wider than the target spacing.
    That is the honest outcome: an empty log bin is dropped rather than filled by
    interpolation.
    """
    truth, chain = make_airframe(), make_chain()
    _, u, y = _sweep(truth, chain, duration_s=60.0)
    est = estimate_frf(u, y, LOG_RATE_HZ, nperseg=4096, input_signal="u", output_signal="y")
    smoothed = log_smooth(est, points_per_decade=30)

    assert smoothed.f_hz.size < est.f_hz.size
    assert smoothed.f_hz.min() >= est.f_hz[est.f_hz > 0].min()
    assert smoothed.f_hz.max() <= est.f_hz.max()

    top = smoothed.f_hz >= 10.0
    spacing = np.diff(np.log10(smoothed.f_hz[top]))
    assert spacing.max() == pytest.approx(1.0 / 30.0, rel=0.2)


def test_valid_band_is_reported_from_the_gate_not_assumed() -> None:
    truth, chain = make_airframe(), make_chain()
    _, u, y = _sweep(truth, chain)
    frf: FrequencyResponse = _effective_plant(u, y).frf
    low, high = frf.valid_band_hz
    assert low >= F_START * 0.8
    assert high <= F_STOP * 1.2
    assert frf.coherence_mean > 0.9
