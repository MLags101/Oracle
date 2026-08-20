"""Why this number? (spec section 8, the teaching layer).

Every recommended value in the GUI carries an affordance that opens the trace
behind it: which model it came from, which band that model is valid over, which
constraint stopped the number going further, and what was rejected on the way.

The rule this module is built on is that **an explanation contains the user's own
numbers or it does not ship**. A paragraph explaining what phase margin means in
general is a textbook; a paragraph saying that *this* design stopped at 6.1 Hz
because the phase margin hit 45 degrees, and that the notch it is carrying costs
9 of those degrees, is an explanation. So every entry here is a function of a
``TuneRecommendation``, not a string constant, and the glossary -- which *is*
general -- is kept separate and linked to rather than mixed in.

Nothing here computes anything. If a number appears in an explanation it was
already decided elsewhere and is being read back; a second derivation of the same
quantity would be a second chance to disagree with the design.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from rotorid.core.filters.harmonic import HarmonicNotch
from rotorid.core.types import Axis, TuneRecommendation

__all__ = [
    "GLOSSARY",
    "Explanation",
    "GlossaryEntry",
    "explain",
    "explainable",
    "glossary_for",
]


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    """A term, defined once, linked to from wherever it is used."""

    term: str
    short: str
    detail: str
    see_also: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Explanation:
    """The trace behind one recommended number.

    Attributes:
        key: Stable identifier, used by the GUI to attach the affordance.
        title: What the number is called where the user sees it.
        value: The number, formatted with the units it is quoted in.
        headline: One sentence: what this quantity does in the loop.
        because: The reasoning, in order, each line carrying a number from this
            analysis.
        binding: What stopped it going further, when something did.
        alternatives: Rejected options and why they lost.
        glossary: Terms in ``GLOSSARY`` used above, for the GUI to link.
    """

    key: str
    title: str
    value: str
    headline: str
    because: tuple[str, ...]
    binding: str | None = None
    alternatives: tuple[tuple[str, str], ...] = ()
    glossary: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Glossary
# --------------------------------------------------------------------------- #

GLOSSARY: dict[str, GlossaryEntry] = {
    "phase_margin": GlossaryEntry(
        term="Phase margin",
        short="How much extra lag the loop can absorb before it oscillates.",
        detail=(
            "At the frequency where the loop gain passes 1, phase margin is how far "
            "the phase is from -180 degrees. At zero the loop sustains an oscillation "
            "on its own. The 45 degree target is the ADS-33 / US Army AFDD rotorcraft "
            "convention; flight-test work found 20-23 degrees already produces "
            "objectionable oscillation and pilot-induced-oscillation tendencies, which "
            "is why this tool refuses to design below 25 degrees at any slider "
            "position. Phase margin is a lag budget, and every filter spends some of it."
        ),
        see_also=("crossover", "delay_margin", "notch_phase"),
    ),
    "gain_margin": GlossaryEntry(
        term="Gain margin",
        short="How much stronger the loop could get before it oscillates.",
        detail=(
            "At the frequency where the loop phase passes -180 degrees, gain margin is "
            "how far below 1 the loop gain is. It covers the things that change after "
            "the log was recorded: a heavier battery, worn props, a different payload. "
            "6 dB means the airframe could double its response before the loop went "
            "unstable."
        ),
        see_also=("phase_margin",),
    ),
    "crossover": GlossaryEntry(
        term="Crossover frequency",
        short="Where loop gain passes 1 -- roughly how fast the loop is.",
        detail=(
            "Below crossover the loop corrects errors; above it, it does not. Raising "
            "crossover makes the aircraft feel sharper and reject gusts harder, and "
            "costs phase margin, because both the airframe's own lag and every filter "
            "in the path contribute more phase the higher you go."
        ),
        see_also=("phase_margin", "drb"),
    ),
    "drb": GlossaryEntry(
        term="Disturbance-rejection bandwidth",
        short="The highest frequency of disturbance the loop still pushes back on.",
        detail=(
            "The lowest frequency at which the sensitivity function reaches -3 dB. "
            "Above it, a gust moves the aircraft essentially unopposed. This is what "
            "the design maximizes, rather than crossover: it is the performance the "
            "pilot actually feels in wind."
        ),
        see_also=("drp", "crossover"),
    ),
    "drp": GlossaryEntry(
        term="Disturbance-rejection peak",
        short="The worst amplification of disturbance, at the frequency it happens.",
        detail=(
            "The peak of the sensitivity function. Every feedback loop amplifies "
            "disturbance somewhere -- that is unavoidable, not a design error -- but a "
            "large peak means a narrow band where the loop makes gusts worse rather "
            "than better, which feels like a twitchy or ringy aircraft."
        ),
        see_also=("drb",),
    ),
    "delay_margin": GlossaryEntry(
        term="Delay margin",
        short="Phase margin restated as milliseconds of extra delay.",
        detail=(
            "Phase margin divided by crossover frequency. Useful because the things "
            "that eat it -- a slower loop rate, a lower gyro cutoff, an added notch, a "
            "different ESC protocol -- are all naturally quoted in milliseconds."
        ),
        see_also=("phase_margin",),
    ),
    "coherence": GlossaryEntry(
        term="Coherence",
        short="How much of the measured output the measured input explains.",
        detail=(
            "Between 0 and 1, per frequency. Low coherence means the identification "
            "is looking at something other than the response to the input -- noise, "
            "wind, or a nonlinearity -- and the model over that band is not evidence. "
            "The airframe is only fitted where coherence passes the gate."
        ),
        see_also=("effective_plant",),
    ),
    "effective_plant": GlossaryEntry(
        term="Effective plant vs. airframe",
        short="What the log shows includes the filters; the airframe does not.",
        detail=(
            "Both firmware stacks log the *filtered* gyro, so the response measured "
            "from a log is the airframe with the flown filter chain already in it. The "
            "airframe on its own is that response with the modelled chain divided back "
            "out. Keeping the two apart is what stops a filter's phase lag being "
            "counted twice -- which would make every recommended gain too timid -- or "
            "not at all, which would make them oscillate."
        ),
        see_also=("coherence", "notch_phase"),
    ),
    "notch_phase": GlossaryEntry(
        term="Notch phase lag",
        short="A notch bends the phase well below the frequency it removes.",
        detail=(
            "A notch is not free and is not local. It costs phase over a band roughly "
            "as wide as its own bandwidth, and that band reaches down toward crossover. "
            "This is why a wide notch, a deep notch and a stack of harmonics are each "
            "paid for in gains you can no longer carry -- and why the filters and the "
            "gains here are solved together rather than one after the other."
        ),
        see_also=("phase_margin", "dterm_noise"),
    ),
    "dterm_noise": GlossaryEntry(
        term="D-term noise",
        short="How much of the motor command is the D gain amplifying noise.",
        detail=(
            "The derivative term multiplies by frequency, so it amplifies exactly the "
            "high-frequency noise the filters are there to remove. Quoted here as the "
            "RMS of the D contribution as a percentage of full motor output, computed "
            "by propagating the measured pre-filter gyro spectrum through the "
            "recommended chain and the recommended D gain. It is what decides how much "
            "D a real aircraft can carry, and it is why filters and gains cannot be "
            "chosen independently."
        ),
        see_also=("notch_phase",),
    ),
    "conservatism": GlossaryEntry(
        term="Conservatism",
        short="How much margin the design holds back beyond the minimum.",
        detail=(
            "0 designs to the stated margin targets. 1 raises the phase-margin target "
            "well above them, giving up bandwidth for tolerance to everything the log "
            "did not show: a different battery, a payload, colder air, worn props."
        ),
        see_also=("phase_margin",),
    ),
}


def glossary_for(explanation: Explanation) -> tuple[GlossaryEntry, ...]:
    """The entries an explanation links to, in the order it used them."""
    return tuple(GLOSSARY[key] for key in explanation.glossary if key in GLOSSARY)


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #

_AP_SUFFIX: dict[Axis, str] = {"roll": "RLL", "pitch": "PIT", "yaw": "YAW"}


def _canonical(key: str, axis: Axis) -> str:
    """Map a stack parameter name onto the key its explanation is filed under.

    The GUI shows ``ATC_RAT_PIT_D``; the explanation is the same one for every
    axis, filled with that axis's numbers.
    """
    suffix = _AP_SUFFIX[axis]
    per_axis = {
        f"ATC_RAT_{suffix}_P": "rate_p",
        f"ATC_RAT_{suffix}_I": "rate_i",
        f"ATC_RAT_{suffix}_D": "rate_d",
        f"ATC_RAT_{suffix}_FF": "rate_ff",
        f"ATC_RAT_{suffix}_FLTD": "dterm_lpf",
        f"ATC_ANG_{suffix}_P": "attitude_p",
    }
    return per_axis.get(key, key)


# --------------------------------------------------------------------------- #
# Explanations
# --------------------------------------------------------------------------- #


def _rate_p(rec: TuneRecommendation) -> Explanation:
    return Explanation(
        key="rate_p",
        title=f"Rate {rec.axis} P",
        value=f"{rec.gains.kp:.4g}",
        headline=(
            "Proportional gain sets how hard the loop pushes back on a rate error "
            "right now, and with the loop shape fixed it is what places the crossover."
        ),
        because=(
            f"Was {rec.baseline_gains.kp:.4g}, now {rec.gains.kp:.4g} "
            f"({_ratio(rec.baseline_gains.kp, rec.gains.kp)}).",
            f"The shape (I/P and D/P) was chosen first, which leaves the loop linear "
            f"in P -- so P alone slides the loop up until it crosses over at "
            f"{rec.margins.crossover_hz:.2f} Hz.",
            f"At that crossover the phase margin is "
            f"{rec.margins.phase_margin_deg:.0f} degrees and the gain margin "
            f"{rec.margins.gain_margin_db:.1f} dB, against an airframe identified "
            f"over {rec.model.valid_band_hz[0]:.2f}-{rec.model.valid_band_hz[1]:.1f} Hz "
            f"at mean coherence {rec.model.coherence_mean:.2f}.",
            f"Pushing P higher moves the crossover up, and "
            f"{_binding_phrase(rec.binding_constraint)} is what it runs into.",
        ),
        binding=rec.binding_constraint,
        glossary=("crossover", "phase_margin", "gain_margin", "coherence"),
    )


def _rate_i(rec: TuneRecommendation) -> Explanation:
    corner = rec.gains.ki / rec.gains.kp if rec.gains.kp > 0.0 else 0.0
    return Explanation(
        key="rate_i",
        title=f"Rate {rec.axis} I",
        value=f"{rec.gains.ki:.4g}",
        headline=(
            "Integral gain removes the steady error P leaves behind -- the lean in "
            "wind, the offset from an imperfect trim."
        ),
        because=(
            f"Was {rec.baseline_gains.ki:.4g}, now {rec.gains.ki:.4g}.",
            f"What was actually chosen is the ratio I/P = {corner:.1f} 1/s, which is "
            f"the frequency below which the integrator takes over. That sits "
            f"{rec.margins.crossover_hz * 6.283 / max(corner, 1e-9):.1f}x below "
            f"crossover, far enough down that it costs the loop almost none of its "
            f"{rec.margins.phase_margin_deg:.0f} degrees of phase margin.",
            "Integral action always costs phase; keeping its corner well below "
            "crossover is how that cost is kept small.",
        ),
        glossary=("crossover", "phase_margin"),
    )


def _rate_d(rec: TuneRecommendation) -> Explanation:
    return Explanation(
        key="rate_d",
        title=f"Rate {rec.axis} D",
        value=f"{rec.gains.kd:.4g}",
        headline=(
            "Derivative gain adds phase lead, which is what buys the phase margin "
            "that lets P be larger -- and it amplifies noise doing it."
        ),
        because=(
            f"Was {rec.baseline_gains.kd:.4g}, now {rec.gains.kd:.4g}.",
            f"D/P = {rec.gains.kd / rec.gains.kp if rec.gains.kp > 0 else 0.0:.4f} s "
            f"places the lead where crossover is, at "
            f"{rec.margins.crossover_hz:.2f} Hz.",
            f"The limit on it is noise, not stability: at this D gain the derivative "
            f"term contributes {rec.dterm_noise_rms_pct:.2f}% of full motor output as "
            f"RMS noise, computed by pushing the measured pre-filter gyro spectrum "
            f"through the recommended filter chain.",
            f"That figure is why the filters and the gains were solved together. The "
            f"chain carries {rec.filters.phase_cost_deg:.1f} degrees of phase at "
            f"crossover; a quieter chain would allow more D but cost more of the "
            f"{rec.margins.phase_margin_deg:.0f} degrees available.",
        ),
        binding=rec.binding_constraint,
        glossary=("dterm_noise", "notch_phase", "phase_margin"),
    )


def _rate_ff(rec: TuneRecommendation) -> Explanation:
    return Explanation(
        key="rate_ff",
        title=f"Rate {rec.axis} FF",
        value=f"{rec.gains.kff:.4g}",
        headline=(
            "Feed-forward sends the commanded rate straight to the motors without "
            "waiting for an error to appear."
        ),
        because=(
            f"Was {rec.baseline_gains.kff:.4g}, now {rec.gains.kff:.4g}.",
            "It sits outside the feedback path, so it changes how the aircraft "
            "follows the stick and changes neither the phase margin nor the gain "
            "margin. Nothing it does can destabilise the loop.",
            f"Predicted step response with it: {rec.predicted_step.rise_time_s * 1000:.0f} ms "
            f"rise, {rec.predicted_step.overshoot_pct:.0f}% overshoot.",
        ),
        glossary=("phase_margin",),
    )


def _attitude_p(rec: TuneRecommendation) -> Explanation:
    return Explanation(
        key="attitude_p",
        title=f"Attitude {rec.axis} P",
        value=f"{6.283 * rec.margins.crossover_hz / 4.0:.2f} 1/s",
        headline=(
            "The outer loop turns an angle error into a rate command. Its only real "
            "constraint is that it must be slower than the rate loop underneath it."
        ),
        because=(
            f"The rate loop now crosses over at {rec.margins.crossover_hz:.2f} Hz "
            f"({6.283 * rec.margins.crossover_hz:.1f} rad/s).",
            "The outer loop is sized a factor of 4 below that, which is the "
            "separation at which the inner loop looks like a simple lag to the outer "
            "one and the two stop interacting.",
            "It is derived from the rate loop that now exists rather than read off a "
            "table, which is why it changes when the rate gains change.",
        ),
        glossary=("crossover",),
    )


def _dterm_lpf(rec: TuneRecommendation) -> Explanation:
    cutoff = rec.filters.chain.dterm_lpf_hz
    return Explanation(
        key="dterm_lpf",
        title=f"D-term low-pass ({rec.axis})",
        value=_hz(cutoff),
        headline=(
            "A low-pass on the derivative branch only. It cuts the noise D amplifies "
            "without touching what P and I see."
        ),
        because=(
            f"Sits below the gyro low-pass at {_hz(rec.filters.chain.gyro_lpf_hz)}, "
            f"because the derivative amplifies with frequency and needs the tighter "
            f"limit.",
            f"At this cutoff the D term contributes {rec.dterm_noise_rms_pct:.2f}% of "
            f"full motor output as RMS noise.",
            "Lower would be quieter and would cost phase where the D lead is supposed "
            "to be doing its work, which would cancel the reason for having D at all.",
        ),
        glossary=("dterm_noise", "notch_phase"),
    )


def _gyro_lpf(rec: TuneRecommendation) -> Explanation:
    return Explanation(
        key="INS_GYRO_FILTER",
        title="Gyro low-pass",
        value=_hz(rec.filters.chain.gyro_lpf_hz),
        headline=(
            "The low-pass every term in the controller sees, so its phase lag is "
            "paid by the whole loop."
        ),
        because=(
            f"Chosen as the highest cutoff on the ladder that still holds the D-term "
            f"noise limit, which it does at {rec.dterm_noise_rms_pct:.2f}% of full "
            f"output.",
            f"Highest, not lowest: the whole chain already costs "
            f"{rec.filters.phase_cost_deg:.1f} degrees at the "
            f"{rec.margins.crossover_hz:.2f} Hz crossover, and every hertz of cutoff "
            f"given away is phase margin given away with it.",
            "Peaks that track the motors belong to the notch, not to this filter. "
            "Dropping the cutoff to chase them costs far more phase than notching "
            "them does.",
        ),
        glossary=("notch_phase", "phase_margin", "dterm_noise"),
    )


def _notch_freq(rec: TuneRecommendation) -> Explanation:
    notch = _first_notch(rec)
    return Explanation(
        key="INS_HNTCH_FREQ",
        title="Notch centre frequency",
        value=_hz(notch.freq_hz if notch else None),
        headline=(
            "The motor fundamental the notch stack is anchored to. Harmonics follow "
            "it as multiples."
        ),
        because=_notch_freq_reasons(rec, notch),
        alternatives=rec.filters.rejected,
        glossary=("notch_phase",),
    )


def _notch_freq_reasons(rec: TuneRecommendation, notch: HarmonicNotch | None) -> tuple[str, ...]:
    if notch is None:
        return ("No notch is recommended: nothing in the spectrum needs one.",)
    reasons = [
        f"Anchored at {notch.freq_hz:.0f} Hz, with harmonics at "
        + ", ".join(f"{notch.freq_hz * h:.0f} Hz" for h in notch.harmonics)
        + ".",
        rec.filters.rationale,
    ]
    if notch.freq_min_ratio < 1.0:
        reasons.append(
            f"Tracking is allowed down to {notch.freq_min_ratio:.2f}x that, i.e. "
            f"{notch.freq_hz * notch.freq_min_ratio:.0f} Hz, so the notch still finds "
            f"the peak at low throttle instead of sitting above it."
        )
    return tuple(reasons)


def _notch_bw(rec: TuneRecommendation) -> Explanation:
    notch = _first_notch(rec)
    ratio = (
        f"{notch.freq_hz / notch.bandwidth_hz:.1f}:1"
        if notch and notch.bandwidth_hz > 0.0
        else "n/a"
    )
    return Explanation(
        key="INS_HNTCH_BW",
        title="Notch bandwidth",
        value=_hz(notch.bandwidth_hz if notch else None),
        headline=(
            "How wide the notch is. Wide enough to hold the peak as it moves, and "
            "no wider, because width is paid for in phase."
        ),
        because=(
            f"Frequency-to-bandwidth ratio {ratio}.",
            "It has to cover the peak's own measured width plus the distance the "
            "motor frequency moves between the throttle settings in this log -- a "
            "notch narrower than the peak's excursion simply misses it part of the "
            "time.",
            f"The whole chain costs {rec.filters.phase_cost_deg:.1f} degrees at the "
            f"{rec.margins.crossover_hz:.2f} Hz crossover, out of the "
            f"{rec.margins.phase_margin_deg:.0f} degrees the design ended up with. "
            f"Widening it further has to be paid for out of that.",
        ),
        alternatives=rec.filters.rejected,
        glossary=("notch_phase", "phase_margin"),
    )


def _notch_att(rec: TuneRecommendation) -> Explanation:
    notch = _first_notch(rec)
    attenuations = rec.filters.attenuation_at_peaks_db
    measured = (
        "; ".join(f"{f:.0f} Hz down {abs(db):.0f} dB" for f, db in sorted(attenuations.items()))
        if attenuations
        else "no peaks required attenuation"
    )
    return Explanation(
        key="INS_HNTCH_ATT",
        title="Notch attenuation",
        value=f"{notch.attenuation_db:.0f} dB" if notch else "-",
        headline="How deep the notch cuts at its centre.",
        because=(
            f"Sized to bring the measured peaks down to the target excess above the "
            f"local noise floor: {measured}.",
            "Deeper is not better. Attenuation and bandwidth are coupled in the "
            "firmware's own filter design, so asking for more depth widens the notch "
            "and costs phase away from the frequency you wanted quiet.",
        ),
        alternatives=rec.filters.rejected,
        glossary=("notch_phase",),
    )


def _notch_mode(rec: TuneRecommendation) -> Explanation:
    mode = rec.filters.params.get("INS_HNTCH_MODE")
    names = {
        0.0: "static (fixed frequency)",
        1.0: "throttle-derived",
        2.0: "RPM sensor",
        3.0: "ESC telemetry",
        4.0: "in-flight FFT",
        5.0: "second RPM sensor",
    }
    return Explanation(
        key="INS_HNTCH_MODE",
        title="Notch tracking source",
        value=names.get(mode, "unchanged") if mode is not None else "unchanged",
        headline=(
            "Where the notch gets the motor frequency from. This is the single "
            "biggest determinant of whether the notch works in flight."
        ),
        because=(
            "Sources are ranked by how directly they measure motor speed -- ESC "
            "telemetry above the in-flight FFT above the throttle model above a fixed "
            "frequency -- and gated on what this log proves the aircraft actually has.",
            rec.filters.rationale,
        ),
        alternatives=rec.filters.rejected,
        glossary=("notch_phase",),
    )


def _notch_hmncs(rec: TuneRecommendation) -> Explanation:
    notch = _first_notch(rec)
    harmonics = ", ".join(f"{h}x" for h in notch.harmonics) if notch else "-"
    return Explanation(
        key="INS_HNTCH_HMNCS",
        title="Notch harmonics",
        value=harmonics,
        headline="Which multiples of the motor frequency get their own notch.",
        because=(
            "A harmonic is included only if its measured peak is large enough that "
            "removing it buys more than the phase the extra notch costs.",
            f"Every included harmonic adds to the "
            f"{rec.filters.phase_cost_deg:.1f} degrees the chain spends at crossover, "
            f"and to the {rec.filters.cpu_cost_rel:.1f} biquad-equivalents of load per "
            f"loop iteration.",
        ),
        alternatives=rec.filters.rejected,
        glossary=("notch_phase",),
    )


def _phase_margin(rec: TuneRecommendation) -> Explanation:
    return Explanation(
        key="phase_margin",
        title="Phase margin",
        value=f"{rec.margins.phase_margin_deg:.0f} deg",
        headline=GLOSSARY["phase_margin"].short,
        because=(
            f"Measured at the {rec.margins.crossover_hz:.2f} Hz crossover of the loop "
            f"built from the recommended gains and the recommended filters together.",
            f"Equivalent to {rec.margins.delay_margin_ms:.0f} ms of extra delay before "
            f"the loop would oscillate.",
            f"The filter chain is already spending "
            f"{rec.filters.phase_cost_deg:.1f} degrees of the budget, and the airframe "
            f"and actuation delay account for "
            f"{rec.latency.airframe_tau_deg + rec.latency.actuator_deg:.1f} more.",
        ),
        binding=rec.binding_constraint,
        glossary=("phase_margin", "crossover", "delay_margin", "notch_phase"),
    )


def _gain_margin(rec: TuneRecommendation) -> Explanation:
    return Explanation(
        key="gain_margin",
        title="Gain margin",
        value=f"{rec.margins.gain_margin_db:.1f} dB",
        headline=GLOSSARY["gain_margin"].short,
        because=(
            f"The airframe would have to respond "
            f"{10 ** (rec.margins.gain_margin_db / 20.0):.1f}x more strongly than it "
            f"did in this log before the loop went unstable.",
            "That is the cover for everything the log did not show: a fresh battery, "
            "a lighter payload, colder air, different props.",
            *_spread_line(rec),
        ),
        binding=rec.binding_constraint,
        glossary=("gain_margin",),
    )


def _crossover(rec: TuneRecommendation) -> Explanation:
    return Explanation(
        key="crossover",
        title="Crossover frequency",
        value=f"{rec.margins.crossover_hz:.2f} Hz",
        headline=GLOSSARY["crossover"].short,
        because=(
            f"Sits inside the band the airframe was identified over "
            f"({rec.model.valid_band_hz[0]:.2f}-{rec.model.valid_band_hz[1]:.1f} Hz), "
            f"so the model is describing the aircraft here rather than extrapolating.",
            f"{_binding_phrase(rec.binding_constraint)} is what stops it going higher.",
        ),
        binding=rec.binding_constraint,
        glossary=("crossover", "coherence", "effective_plant"),
    )


def _drb(rec: TuneRecommendation) -> Explanation:
    return Explanation(
        key="drb",
        title="Disturbance-rejection bandwidth",
        value=f"{rec.margins.disturbance_rejection_bw_hz:.2f} Hz",
        headline=GLOSSARY["drb"].short,
        because=(
            "This is the quantity the design maximizes. Crossover, gains and filters "
            "are all means to it.",
            f"Worst-case amplification along the way is "
            f"{rec.margins.disturbance_rejection_peak_db:.1f} dB.",
            f"{_binding_phrase(rec.binding_constraint)} is the constraint it is sitting on.",
        ),
        binding=rec.binding_constraint,
        glossary=("drb", "drp"),
    )


def _dterm_noise(rec: TuneRecommendation) -> Explanation:
    return Explanation(
        key="dterm_noise",
        title="D-term noise",
        value=f"{rec.dterm_noise_rms_pct:.2f}% of full output",
        headline=GLOSSARY["dterm_noise"].short,
        because=(
            f"Computed by propagating the measured pre-filter gyro spectrum through "
            f"the recommended chain ({rec.filters.chain.describe()}) and the "
            f"recommended D gain of {rec.gains.kd:.4g}.",
            "It is a ceiling, not a target: the D gain was raised until this figure "
            "reached its limit or the margins did, whichever came first.",
        ),
        glossary=("dterm_noise", "notch_phase"),
    )


def _confidence(rec: TuneRecommendation) -> Explanation:
    return Explanation(
        key="confidence",
        title="Confidence",
        value=rec.confidence,
        headline=(
            "How much the identification behind these numbers deserves to be "
            "trusted -- which is about the evidence, not about the fit."
        ),
        because=(
            f"Identified over {rec.model.valid_band_hz[0]:.2f}-"
            f"{rec.model.valid_band_hz[1]:.1f} Hz at mean coherence "
            f"{rec.model.coherence_mean:.2f}.",
            f"Fit residual {rec.model.fit_rms_db:.2f} dB and "
            f"{rec.model.fit_rms_deg:.1f} degrees against a {rec.model.structure} "
            f"structure.",
            f"Filter chain divided out by the {rec.model.filter_deconvolution} route.",
            "A model can fit a narrow, weakly excited band beautifully and still "
            "describe the aircraft badly, so band width and coherence count for more "
            "here than residual does.",
        ),
        glossary=("coherence", "effective_plant"),
    )


_REGISTRY: dict[str, Callable[[TuneRecommendation], Explanation]] = {
    "rate_p": _rate_p,
    "rate_i": _rate_i,
    "rate_d": _rate_d,
    "rate_ff": _rate_ff,
    "attitude_p": _attitude_p,
    "dterm_lpf": _dterm_lpf,
    "INS_GYRO_FILTER": _gyro_lpf,
    "INS_HNTCH_FREQ": _notch_freq,
    "INS_HNTCH_BW": _notch_bw,
    "INS_HNTCH_ATT": _notch_att,
    "INS_HNTCH_MODE": _notch_mode,
    "INS_HNTCH_HMNCS": _notch_hmncs,
    "phase_margin": _phase_margin,
    "gain_margin": _gain_margin,
    "crossover": _crossover,
    "drb": _drb,
    "dterm_noise": _dterm_noise,
    "confidence": _confidence,
}


def explain(key: str, rec: TuneRecommendation) -> Explanation | None:
    """The trace behind one number, filled with this recommendation's values.

    Args:
        key: Either a metric key (``"phase_margin"``) or a stack parameter name
            as the user sees it (``"ATC_RAT_PIT_D"``).

    Returns:
        ``None`` when nothing is filed under that key. The caller should then
        show no affordance at all rather than an empty one: a "why?" link that
        opens onto nothing is worse than no link.
    """
    fn = _REGISTRY.get(_canonical(key, rec.axis))
    return fn(rec) if fn is not None else None


def explainable(rec: TuneRecommendation) -> tuple[str, ...]:
    """Every key that has an explanation for this recommendation.

    Filters that were left alone are omitted: there is no notch bandwidth to
    explain when no notch was recommended, and no D-term noise figure to explain
    when the log was too slow to measure one.
    """
    notch = _first_notch(rec)
    keys = []
    for key in _REGISTRY:
        if key.startswith("INS_HNTCH_") and notch is None:
            continue
        if key == "INS_GYRO_FILTER" and rec.filters.chain.gyro_lpf_hz is None:
            continue
        if key == "dterm_lpf" and rec.filters.chain.dterm_lpf_hz is None:
            continue
        # A log too slow to carry a noise spectrum leaves this NaN. An
        # explanation of a number that was never measured is not an explanation.
        if key == "dterm_noise" and not math.isfinite(rec.dterm_noise_rms_pct):
            continue
        keys.append(key)
    return tuple(keys)


# --------------------------------------------------------------------------- #
# Small shared pieces
# --------------------------------------------------------------------------- #

_BINDING_PHRASE = {
    "phase_margin": "the phase margin target",
    "gain_margin": "the gain margin target",
    "peak_sensitivity": "the sensitivity peak limit",
    "identified_band": "the top of the identified band -- beyond it the model is extrapolating",
    "delay_limit": "the delay in the airframe and actuation",
}


def _binding_phrase(constraint: str) -> str:
    return _BINDING_PHRASE.get(constraint, constraint.replace("_", " "))


def _first_notch(rec: TuneRecommendation) -> HarmonicNotch | None:
    """The fundamental notch stack, or None if no notch is recommended."""
    return rec.filters.chain.notches[0] if rec.filters.chain.notches else None


def _hz(value: float | None) -> str:
    return f"{value:.0f} Hz" if value else "off"


def _ratio(old: float, new: float) -> str:
    if old <= 0.0:
        return "up from zero"
    factor = new / old
    if abs(factor - 1.0) < 0.01:
        return "unchanged"
    return f"{factor:.2f}x" if factor > 1.0 else f"down to {factor:.2f}x"


def _spread_line(rec: TuneRecommendation) -> tuple[str, ...]:
    """The measured gain spread, when there was more than one operating point."""
    spread = rec.model.gain_spread_pct
    if spread is None:
        return ()
    return (
        f"The airframe gain itself moved {spread:.0f}% across the operating points in "
        f"this log, which is the part of that cover already spent.",
    )
