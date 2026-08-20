"""The predicted step response.

Written after finding that :func:`step_response` returned a flat line of exactly
1.0 for every controller and every aircraft, and had done since it was written.
The cause was arithmetic that reads correctly at a glance: multiply the transform
of the input by the transfer function and invert. The transform of a *constant*
sequence is a single spike at DC, so the product keeps ``T(0)`` and discards the
entire response -- and ``T(0)`` is set to 1.0 whenever there is an integrator.

Nothing caught it. The report printed "0 ms rise, 0% overshoot", the Design stage
drew a horizontal line at 1.0, and every test asserted only that the numbers
existed. So these tests assert on the *shape*: that the response starts at zero,
that it goes somewhere, and that it moves in the direction the control theory says
it should when a gain is changed.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotorid.core.analysis.margins import loop_delay
from rotorid.core.analysis.step import step_metrics, step_response
from rotorid.core.design.controller import controller_for
from rotorid.core.filters.chain import OperatingPoint
from rotorid.core.types import GainSet
from tests.synthetic.generators import make_airframe, make_chain

AIRFRAME = make_airframe()
CHAIN = make_chain()
OP = OperatingPoint(motor_hz=(50.0,))
DELAY = loop_delay(loop_rate_hz=400.0, actuator_ms=0.1, zoh_loops=0.5, compute_loops=1.0)


def _step(kp: float = 0.135, ki: float = 0.135, kd: float = 0.0036, duration_s: float = 2.0):
    controller = controller_for(
        "ardupilot",
        GainSet(
            axis="roll",
            kp=kp,
            ki=ki,
            kd=kd,
            kff=0.0,
            dterm_lpf_hz=CHAIN.dterm_lpf_hz,
            error_lpf_hz=CHAIN.error_lpf_hz,
            target_lpf_hz=CHAIN.target_lpf_hz,
        ),
        CHAIN,
    )
    return step_response(
        controller, AIRFRAME, delay=DELAY, op=OP, duration_s=duration_s, sample_rate_hz=800.0
    )


def test_the_response_is_not_a_constant() -> None:
    """The regression. A flat line is what this looked like for its whole life."""
    _, y = _step()
    assert float(np.ptp(y)) > 0.5, "the step response does not move"


def test_the_aircraft_starts_where_it_started() -> None:
    """A rate response to a step command begins at zero, not at the answer."""
    _, y = _step()
    assert abs(float(y[0])) < 0.05
    # Not all the way to 1.0 in two seconds -- the integrator is still climbing.
    assert float(np.max(y)) > 0.85


def test_nothing_happens_before_the_command() -> None:
    """A circular inverse FFT wraps the undecayed tail onto the start.

    Which reads as an aircraft that begins responding before it is asked, and is
    the reason the transfer function is evaluated over four times the span kept.
    """
    _, y = _step()
    assert float(np.max(np.abs(y[:8]))) < 0.05


def test_more_proportional_gain_is_faster_and_rings_more() -> None:
    """The one relationship every user already knows. If it is backwards, so is
    everything the Design stage teaches."""
    soft = step_metrics(*_step(kp=0.09))
    stiff = step_metrics(*_step(kp=0.27))
    assert stiff.rise_time_s < soft.rise_time_s
    assert stiff.overshoot_pct > soft.overshoot_pct


def test_an_integrator_removes_the_steady_state_error() -> None:
    """Given long enough. The default two-second window is not long enough, which
    is why the recommendation computes its metrics over four."""
    _, with_i = _step(ki=0.135, duration_s=12.0)
    _, without_i = _step(ki=0.0, duration_s=12.0)
    assert float(with_i[-1]) == pytest.approx(1.0, abs=0.01)
    assert float(without_i[-1]) < 0.95


def test_the_response_converges_rather_than_jumping_to_its_answer() -> None:
    """Monotone approach to unity over lengthening windows -- an integrator with a
    one-second time constant has not finished inside the window a pilot judges by,
    and a metric read off too short a record reports an error the tune lacks."""
    finals = [float(_step(duration_s=d)[1][-1]) for d in (1.0, 2.0, 5.0, 10.0)]
    assert finals == sorted(finals)
    assert finals[0] < 0.9 < finals[-1]
