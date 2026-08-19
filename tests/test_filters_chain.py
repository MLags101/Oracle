"""Tests for filter chain assembly, evaluation and the latency budget."""

from __future__ import annotations

import numpy as np
import pytest

from rotorid.config import load_config
from rotorid.core.filters.chain import FilterChain, OperatingPoint, ardupilot_chain
from rotorid.core.filters.harmonic import HarmonicNotch, NotchOption
from rotorid.core.filters.latency import actuator_latency_ms, build_budget, delay_phase_deg

GYRO_RATE = 4000.0
LOOP_RATE = 400.0

AP_PARAMS: dict[str, float] = {
    "INS_GYRO_FILTER": 42.0,
    "INS_HNTCH_ENABLE": 1.0,
    "INS_HNTCH_MODE": 3.0,
    "INS_HNTCH_FREQ": 80.0,
    "INS_HNTCH_BW": 40.0,
    "INS_HNTCH_ATT": 40.0,
    "INS_HNTCH_HMNCS": 3.0,  # fundamental + 2nd
    "INS_HNTCH_REF": 1.0,
    "INS_HNTCH_FM_RAT": 1.0,
    "INS_HNTCH_OPTS": 0.0,
    "ATC_RAT_RLL_FLTD": 21.0,
    "ATC_RAT_RLL_FLTE": 0.0,
    "ATC_RAT_RLL_FLTT": 20.0,
    "MOT_PWM_TYPE": 6.0,  # DShot600
}


def _chain(**overrides: object) -> FilterChain:
    params = dict(AP_PARAMS)
    params.update({k: float(v) for k, v in overrides.items()})  # type: ignore[arg-type]
    return ardupilot_chain(params, "roll", gyro_sample_rate_hz=GYRO_RATE, loop_rate_hz=LOOP_RATE)


# --------------------------------------------------------------------------- #
# Assembly from parameters
# --------------------------------------------------------------------------- #


def test_ardupilot_chain_reads_every_stage() -> None:
    chain = _chain()
    assert chain.stack == "ardupilot"
    assert chain.gyro_lpf_hz == 42.0
    assert chain.dterm_lpf_hz == 21.0
    assert chain.target_lpf_hz == 20.0
    assert chain.error_lpf_hz is None  # FLTE = 0 means disabled, not "default"
    assert len(chain.notches) == 1
    assert chain.notches[0].harmonics == (1, 2)


def test_disabled_notch_is_not_built() -> None:
    assert _chain(INS_HNTCH_ENABLE=0).notches == ()


def test_second_notch_bank_is_read() -> None:
    chain = _chain(
        INS_HNTC2_ENABLE=1,
        INS_HNTC2_FREQ=120.0,
        INS_HNTC2_BW=20.0,
        INS_HNTC2_ATT=30.0,
        INS_HNTC2_HMNCS=1,
    )
    assert len(chain.notches) == 2
    assert chain.notches[1].freq_hz == 120.0


def test_missing_parameters_are_absent_not_guessed() -> None:
    """A parameter that was never logged must not acquire a plausible default."""
    chain = ardupilot_chain({}, "roll", gyro_sample_rate_hz=GYRO_RATE, loop_rate_hz=LOOP_RATE)
    assert chain.gyro_lpf_hz is None
    assert chain.dterm_lpf_hz is None
    assert chain.notches == ()
    assert np.allclose(np.abs(chain.sensor_response(np.array([10.0, 100.0]))), 1.0)


# --------------------------------------------------------------------------- #
# The separation that prevents double counting
# --------------------------------------------------------------------------- #


def test_sensor_response_excludes_pid_local_filters() -> None:
    """sensor_response is gyro LPF + notches only.

    If FLTD or FLTE leaked in here they would be applied twice -- once by the chain
    and once by the controller model -- and every recommendation would come out
    too conservative. This is the guard for that.
    """
    f = np.array([5.0, 10.0, 20.0])
    with_pid = _chain(ATC_RAT_RLL_FLTD=10.0, ATC_RAT_RLL_FLTE=5.0)
    without_pid = _chain(ATC_RAT_RLL_FLTD=0.0, ATC_RAT_RLL_FLTE=0.0)
    assert np.allclose(with_pid.sensor_response(f), without_pid.sensor_response(f))


def test_pid_local_filters_are_reachable_separately() -> None:
    f = np.array([10.0])
    chain = _chain(ATC_RAT_RLL_FLTE=5.0)
    assert np.abs(chain.dterm_lpf_response(f)[0]) < 1.0
    assert np.abs(chain.error_lpf_response(f)[0]) < 1.0
    assert np.abs(chain.target_lpf_response(f)[0]) < 1.0


def test_disabled_pid_filters_are_unity() -> None:
    f = np.array([1.0, 50.0])
    chain = _chain(ATC_RAT_RLL_FLTD=0.0, ATC_RAT_RLL_FLTE=0.0, ATC_RAT_RLL_FLTT=0.0)
    for resp in (
        chain.dterm_lpf_response(f),
        chain.error_lpf_response(f),
        chain.target_lpf_response(f),
    ):
        assert np.allclose(np.abs(resp), 1.0)


def test_pid_filters_are_designed_at_loop_rate_not_gyro_rate() -> None:
    """FLTD runs in the PID at the loop rate; designing it at 4 kHz understates its lag."""
    f = np.array([20.0])
    slow_loop = _chain()
    fast_loop = ardupilot_chain(
        AP_PARAMS, "roll", gyro_sample_rate_hz=GYRO_RATE, loop_rate_hz=1600.0
    )
    assert np.angle(slow_loop.dterm_lpf_response(f)[0]) != pytest.approx(
        np.angle(fast_loop.dterm_lpf_response(f)[0])
    )


# --------------------------------------------------------------------------- #
# Operating point
# --------------------------------------------------------------------------- #


def test_measured_motor_frequency_moves_the_notch() -> None:
    chain = _chain()
    f = np.array([80.0])
    at_80 = np.abs(chain.sensor_response(f, OperatingPoint(motor_hz=(80.0,)))[0])
    at_140 = np.abs(chain.sensor_response(f, OperatingPoint(motor_hz=(140.0,)))[0])
    assert at_80 < at_140, "notch tracked away from 80 Hz, so 80 Hz is less attenuated"


def test_per_motor_option_uses_every_motor_frequency() -> None:
    chain = _chain(INS_HNTCH_OPTS=NotchOption.MULTI_SOURCE, INS_HNTCH_HMNCS=1)
    op = OperatingPoint(motor_hz=(78.0, 82.0, 86.0, 90.0))
    assert chain.n_biquads(op) == 1 + 4  # gyro LPF + one notch per motor


def test_single_source_uses_only_the_first_motor_frequency() -> None:
    chain = _chain(INS_HNTCH_HMNCS=1)
    op = OperatingPoint(motor_hz=(78.0, 82.0, 86.0, 90.0))
    assert chain.n_biquads(op) == 2  # gyro LPF + one notch


def test_throttle_tracking_used_when_no_measured_frequency() -> None:
    chain = _chain(INS_HNTCH_REF=0.35, INS_HNTCH_FM_RAT=0.5, INS_HNTCH_HMNCS=1)
    f = np.array([80.0])
    at_hover = np.abs(chain.sensor_response(f, OperatingPoint(throttle=0.35))[0])
    at_full = np.abs(chain.sensor_response(f, OperatingPoint(throttle=1.0))[0])
    assert at_hover < at_full


def test_no_operating_point_falls_back_to_configured_frequency() -> None:
    chain = _chain(INS_HNTCH_HMNCS=1)
    f = np.array([80.0])
    assert np.abs(chain.sensor_response(f)[0]) == pytest.approx(
        np.abs(chain.sensor_response(f, OperatingPoint(motor_hz=(80.0,)))[0])
    )


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #


def test_group_delay_is_positive_below_the_notch() -> None:
    chain = _chain()
    f = np.linspace(5.0, 40.0, 32)
    assert np.all(chain.group_delay_ms(f) > 0.0)


def test_group_delay_needs_two_points() -> None:
    with pytest.raises(ValueError, match="at least two frequencies"):
        _chain().group_delay_ms(np.array([10.0]))


def test_cpu_cost_increases_with_notch_count() -> None:
    one = _chain(INS_HNTCH_HMNCS=1)
    three = _chain(INS_HNTCH_HMNCS=7)
    triple = _chain(INS_HNTCH_HMNCS=7, INS_HNTCH_OPTS=NotchOption.TRIPLE_NOTCH)
    assert one.cpu_cost() < three.cpu_cost() < triple.cpu_cost()


def test_describe_mentions_every_configured_stage() -> None:
    text = _chain().describe()
    assert "gyro LPF 42 Hz" in text
    assert "notch 80 Hz" in text
    assert "D LPF 21 Hz" in text


def test_describe_says_so_when_nothing_is_configured() -> None:
    chain = ardupilot_chain({}, "roll", gyro_sample_rate_hz=GYRO_RATE, loop_rate_hz=LOOP_RATE)
    assert chain.describe() == "no filtering"


# --------------------------------------------------------------------------- #
# Latency budget
# --------------------------------------------------------------------------- #


def _budget(f_hz: float = 20.0, **chain_overrides: object):
    config = load_config()
    chain = _chain(**chain_overrides)
    return build_budget(
        f_hz,
        chain=chain,
        airframe_tau_s=0.020,
        actuator_ms=actuator_latency_ms(AP_PARAMS, config.section("design")["actuator_latency_ms"]),
        zoh_loops=config.float_("design", "zoh_delay_loops"),
        compute_loops=config.float_("design", "compute_delay_loops"),
        op=OperatingPoint(motor_hz=(80.0,)),
    )


def test_budget_items_partition_the_sensor_response() -> None:
    """gyro LPF + notches must reconstruct the sensor path exactly, with no overlap."""
    chain = _chain()
    op = OperatingPoint(motor_hz=(80.0,))
    budget = _budget()
    sensor_deg = float(chain.phase_deg(np.array([20.0]), op)[0])
    assert budget.gyro_lpf_deg + budget.notches_deg == pytest.approx(sensor_deg, abs=1e-9)


def test_budget_total_excludes_nothing_and_common_path_excludes_dterm() -> None:
    budget = _budget()
    assert budget.total_deg == pytest.approx(budget.common_path_deg + budget.dterm_lpf_deg)
    assert budget.dterm_lpf_deg > 0.0


def test_dterm_filter_costs_more_phase_than_the_whole_notch_stack() -> None:
    """21 Hz FLTD at a 400 Hz loop out-costs a two-harmonic notch stack.

    This is what motivates optimizing the D-term cutoff instead of accepting the
    gyro_filter/2 rule of thumb: the notch usually is not the expensive part.
    """
    budget = _budget()
    assert budget.dterm_lpf_deg > budget.notches_deg
    assert budget.gyro_lpf_deg > budget.notches_deg


def test_transport_delay_can_dominate_the_whole_budget() -> None:
    """A 20 ms airframe delay is 144 degrees at 20 Hz -- more than every filter.

    Delay grows linearly with frequency while filter lag saturates, so on a
    laggy airframe no amount of filter tuning buys a high crossover. The tool has
    to be able to say that, which means the budget must show it.
    """
    budget = _budget()
    assert budget.airframe_tau_deg == pytest.approx(144.0)
    assert budget.airframe_tau_deg > budget.total_deg - budget.airframe_tau_deg


def test_delay_terms_are_linear_in_frequency() -> None:
    low, high = _budget(10.0), _budget(20.0)
    assert high.zoh_deg == pytest.approx(2.0 * low.zoh_deg)
    assert high.airframe_tau_deg == pytest.approx(2.0 * low.airframe_tau_deg)


def test_delay_phase_matches_360_f_t() -> None:
    assert delay_phase_deg(0.005, 20.0) == pytest.approx(36.0)


def test_actuator_latency_from_pwm_type() -> None:
    table = load_config().section("design")["actuator_latency_ms"]
    assert actuator_latency_ms({"MOT_PWM_TYPE": 6.0}, table) == table["dshot"]
    assert actuator_latency_ms({"MOT_PWM_TYPE": 0.0}, table) == table["pwm"]
    assert actuator_latency_ms({}, table) == table["unknown"]


def test_budget_rejects_non_positive_frequency() -> None:
    with pytest.raises(ValueError, match="positive frequency"):
        _budget(0.0)


def test_wider_notch_costs_more_lag_in_the_budget() -> None:
    narrow = _budget(20.0, INS_HNTCH_BW=20.0)
    wide = _budget(20.0, INS_HNTCH_BW=40.0)
    assert wide.notches_deg > narrow.notches_deg


# --------------------------------------------------------------------------- #
# Direct construction
# --------------------------------------------------------------------------- #


def test_chain_can_be_built_directly_for_candidate_designs() -> None:
    """The designer proposes chains that no parameter file produced yet."""
    chain = FilterChain(
        stack="ardupilot",
        sample_rate_hz=GYRO_RATE,
        loop_rate_hz=LOOP_RATE,
        gyro_lpf_hz=60.0,
        notches=(
            HarmonicNotch(
                freq_hz=90.0,
                bandwidth_hz=22.5,
                attenuation_db=30.0,
                harmonics=(1, 2),
                sample_rate_hz=GYRO_RATE,
            ),
        ),
        dterm_lpf_hz=30.0,
    )
    assert chain.n_biquads() == 3
    assert "notch 90 Hz BW 22.5" in chain.describe()
