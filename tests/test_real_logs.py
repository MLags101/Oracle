"""Whatever real ArduPilot logs are sitting in ``logs/``, read end to end.

Synthetic bundles test the arithmetic. They cannot test the *reading*, because
they are built out of the same assumptions the reader is written against, so a
reader that misunderstands a message agrees with them perfectly. Every defect
this file exists to catch was found by pointing the tool at a real vehicle's
``.bin`` and looking at what came out.

The directory is gitignored and may well be empty, so every test here skips
rather than fails when there is nothing to read: a developer without the user's
logs should still get a green suite.

Opt in with ``pytest -m real_log``. These are deselected from the default run
because a modern ArduPilot log with raw IMU logging enabled runs to hundreds of
megabytes and takes minutes to parse, and a suite nobody will sit through is a
suite nobody runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rotorid import __version__
from rotorid.config import load_config
from rotorid.core.io.ardupilot import read_ardupilot
from rotorid.core.pipeline import analyze
from rotorid.core.types import LogBundle

_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
_LOGS = sorted(_LOG_DIR.glob("*.bin")) if _LOG_DIR.is_dir() else []

pytestmark = [
    pytest.mark.real_log,
    pytest.mark.skipif(not _LOGS, reason="no real logs in logs/"),
]


@pytest.fixture(scope="module", params=[p.name for p in _LOGS])
def bundle(request: pytest.FixtureRequest) -> LogBundle:
    return read_ardupilot(_LOG_DIR / request.param)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def test_the_reader_gets_through_a_real_log(bundle: LogBundle) -> None:
    assert bundle.stack == "ardupilot"
    assert bundle.firmware_version, "a real log always identifies its firmware"
    assert bundle.params, "a real log always carries PARM messages"
    assert any(k.startswith("rate.") for k in bundle.signals)


def test_provenance_is_a_message_name_and_not_a_unit(bundle: LogBundle) -> None:
    """``source_msg`` used to be filled in with the unit string.

    Nothing displayed it, so nothing noticed, until a finding tried to tell the
    user which message to log faster and said "deg/s was logged at 10 Hz".
    """
    for key, signal in bundle.signals.items():
        if not signal.source_msg:
            continue
        assert "." in signal.source_msg, f"{key}: {signal.source_msg!r} is not MSG.Field"
        assert signal.source_msg not in ("deg/s", "rad/s", "us", "%"), key


def test_every_signal_knows_how_fast_it_was_actually_logged(bundle: LogBundle) -> None:
    """Without this the grid rate is the only rate anything downstream sees."""
    for key, signal in bundle.signals.items():
        assert signal.native_rate_hz is not None, key
        assert signal.native_rate_hz > 0.0, key
        # The grid is built from the vehicle's gyro and loop rates and is
        # normally faster than any message on it. It is never slower.
        assert signal.native_rate_hz <= bundle.sample_rate_hz * 1.01, key


# --------------------------------------------------------------------------- #
# Answering
# --------------------------------------------------------------------------- #


def test_the_pipeline_reaches_a_verdict_on_every_real_log(bundle: LogBundle) -> None:
    """Either a recommendation or a stated reason -- never a silent nothing.

    Two of the three logs this was written against cannot be identified at all.
    That is a fine outcome; producing no findings and no explanation was not.
    """
    result = analyze(bundle, ("roll", "pitch", "yaw"), load_config(), tool_version=__version__)
    assert result.session.recommendations or result.failures
    assert result.session.findings, "a log the tool could not use must say why"


def test_a_log_that_cannot_be_used_still_names_what_to_change(bundle: LogBundle) -> None:
    result = analyze(bundle, ("roll", "pitch", "yaw"), load_config(), tool_version=__version__)
    if result.session.recommendations:
        pytest.skip("this log produced a recommendation")
    assert all(f.action for f in result.session.findings)


def test_nothing_is_recommended_above_the_band_the_log_can_see(bundle: LogBundle) -> None:
    """The failure this whole guard exists for, checked on the real thing.

    A model fitted past the logged Nyquist is fitted to the resampling spline,
    and on a 10 Hz log it comes back looking entirely convincing.
    """
    result = analyze(bundle, ("roll", "pitch", "yaw"), load_config(), tool_version=__version__)
    for axis, rec in result.session.recommendations.items():
        native = bundle.signals[f"rate.{axis}.measured"].native_nyquist_hz
        assert native is not None
        assert rec.model.valid_band_hz[1] <= native * 1.01, axis
