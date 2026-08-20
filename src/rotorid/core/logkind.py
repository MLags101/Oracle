"""What kind of flight this log is, and what that lets the tool do (spec 5.2).

Two logs of the same aircraft can support completely different analyses. A
deliberate SYSTEMID sweep excites a known band on one axis at a time with a
signal the controller did not choose, which is what makes a wide-band airframe
fit trustworthy. Ordinary flight excites whatever the pilot happened to ask for,
in a narrow band, correlated across axes -- but it covers a whole envelope of
throttle settings, battery states and payload conditions that a two-minute sweep
never visits.

Neither is a degraded version of the other, so the tool stops guessing and asks.
The declaration then decides three things that were previously implicit:

* **Which segments count.** A log declared as a tuning flight is identified from
  its deliberate excitation only; falling back to stick inputs would quietly
  turn the answer into a different, weaker answer under the same label.
* **How far the result is allowed to be trusted.** Ordinary flight cannot reach
  ``high`` confidence no matter how well the model happens to fit, because the
  fit residual is measuring agreement over a band the pilot chose.
* **Which analyses are offered at all.** Operating-point sensitivity needs a
  spread of throttle and voltage, which is exactly what a general flight has and
  a hover-and-sweep does not.

The declaration is the user's, not ours. :func:`detect_kind` says what the log
looks like so the two can be compared and a mismatch reported, but a log that
carries a chirp the user never meant to fly is still theirs to describe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rotorid.core.types import AXES, Confidence, LogBundle, LogKind

__all__ = [
    "ANALYSES",
    "KINDS",
    "Capabilities",
    "capabilities",
    "detect_kind",
    "kind_evidence",
]

#: Declaration order, for menus and for ``--kind`` help text.
KINDS: tuple[LogKind, ...] = ("general", "tuning")


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What one kind of log unlocks, and what it caps.

    A single object rather than scattered ``if kind == ...`` tests, because the
    Load screen has to *show* this before any analysis runs. A capability the
    user cannot read before spending two minutes on a run is not a capability,
    it is a surprise.

    Attributes:
        max_confidence: Ceiling applied to every axis, after the evidence-based
            rating is computed. Never a floor -- a bad sweep is still low.
        conservatism_floor: Least docile the designer is allowed to be. Ordinary
            flight identifies a narrower band, so the design has less evidence
            about where the loop actually crosses over.
        offers: Analyses this kind supports, keyed by the short name the Load
            screen and the report both use.
        limits: Plain-language statements of what is *not* available, shown to
            the user rather than discovered by them.
    """

    kind: LogKind
    label: str
    summary: str
    max_confidence: Confidence
    conservatism_floor: float
    offers: tuple[str, ...]
    limits: tuple[str, ...]

    def allows(self, analysis: str) -> bool:
        """Whether one named analysis is available under this declaration."""
        return analysis in self.offers


#: Everything an analysis can be gated on. Named here so a typo in a call to
#: :meth:`Capabilities.allows` is findable rather than silently False.
ANALYSES: tuple[str, ...] = (
    "airframe_id",
    "gain_design",
    "filter_design",
    "noise",
    "oscillation",
    "step_response",
    "operating_point",
)

_TUNING = Capabilities(
    kind="tuning",
    label="Tuning flight",
    summary=(
        "A flight flown to be identified from: a SYSTEMID sweep, or an autotune run. "
        "One axis excited at a time, over a known band, by a signal the controller "
        "did not choose."
    ),
    max_confidence="high",
    conservatism_floor=0.0,
    offers=(
        "airframe_id",
        "gain_design",
        "filter_design",
        "noise",
        "oscillation",
        "step_response",
    ),
    limits=(
        "Operating-point sensitivity is not available: a sweep is flown at one "
        "throttle and one battery state, so the spread of K across the envelope "
        "cannot be measured from it.",
    ),
)

_GENERAL = Capabilities(
    kind="general",
    label="General flight",
    summary=(
        "An ordinary flight. Identification comes from whatever the pilot happened "
        "to excite, so the model is fitted over a narrow band -- but the flight "
        "covers a range of throttle and battery states a sweep never visits."
    ),
    max_confidence="medium",
    conservatism_floor=0.6,
    offers=(
        "airframe_id",
        "gain_design",
        "filter_design",
        "noise",
        "oscillation",
        "step_response",
        "operating_point",
    ),
    limits=(
        "Confidence is capped at medium however well the model fits: the fit is "
        "measuring agreement over a band the pilot chose, not over the band the "
        "loop is designed across.",
        "The designer is held to at least 0.6 conservatism, because a narrow "
        "identification band says less about where the loop crosses over.",
        "A filter change is still designed, but the phase it costs is charged "
        "against a crossover estimated from less evidence.",
    ),
)

_BY_KIND: dict[LogKind, Capabilities] = {"tuning": _TUNING, "general": _GENERAL}


def capabilities(kind: LogKind) -> Capabilities:
    """What the named kind of log supports."""
    try:
        return _BY_KIND[kind]
    except KeyError:
        raise ValueError(f"unknown log kind {kind!r}; expected one of {KINDS}") from None


def detect_kind(bundle: LogBundle) -> LogKind:
    """What this log looks like, ignoring what anybody declared.

    Used to check a declaration rather than to replace one. Deliberate excitation
    is unmistakable when it is there -- an injected chirp is recorded as its own
    signal, and an autotune run announces itself in the event log -- so the test
    is for its presence, and everything else is a general flight.
    """
    return "tuning" if kind_evidence(bundle) else "general"


def kind_evidence(bundle: LogBundle) -> tuple[str, ...]:
    """Why :func:`detect_kind` said what it said, in the user's words.

    Empty means no deliberate excitation was found, which is the whole content of
    the "this looks like a general flight" verdict. Non-empty is quotable back at
    the user next to their declaration, which is what makes a mismatch warning
    actionable instead of merely contradictory.

    Only *recorded* excitation counts. ``SID_AXIS`` deliberately does not: it says
    which axis a sweep would be injected into if one were run, and a log full of
    ordinary hovering with that parameter left set from last week is an ordinary
    flight. What makes a tuning flight is the sweep in the file, not the intent in
    the parameters.
    """
    found: list[str] = []
    for axis in AXES:
        key = f"excite.{axis}"
        signal = bundle.signals.get(key)
        if signal is not None and signal.y.size and bool(np.any(np.abs(signal.y) > 1e-6)):
            found.append(f"an injected SYSTEMID chirp on {axis} ({key})")
    gate = bundle.signals.get("mode.autotune")
    if gate is not None and gate.t.size > 1 and bool(np.any(gate.y > 0.5)):
        seconds = float(np.count_nonzero(gate.y > 0.5)) / max(gate.rate_hz, 1e-9)
        found.append(f"{seconds:.0f} s of the firmware's own autotune ({gate.source_msg})")
    return tuple(found)
