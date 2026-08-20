"""The validation report: two flights, side by side (spec sections 5.10 and 7).

Same constraints as the session report -- one self-contained file, no assets, no
JavaScript, inline SVG -- and the same stylesheet, because the two documents end
up in the same folder and a reader who has learned to read one should not have to
relearn the other.

What differs is the argument. The session report argues from a model and has to
work to stay honest about it. This one argues from two flights, and its whole job
is to keep three different claims from being mistaken for each other: the
aircraft changed, the aircraft improved, and the tool was right. Only the third
is validation, and it is the only one that needs a saved session to make.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from datetime import UTC
from pathlib import Path

import numpy as np

from rotorid.core.analysis.compare import AxisComparison, ValidationReport
from rotorid.core.export.report import STYLE, findings_section, table
from rotorid.core.guidance.validation import validation_findings
from rotorid.core.types import FloatArray, MeasuredStep

__all__ = ["write_comparison"]

#: Figure geometry, matching the session report's plots.
_W, _H = 720, 260
_PAD = 46

#: A data value to an SVG coordinate. The figures build these as closures over
#: their own axis limits, so the drawing helpers take them rather than the limits.
Scale = Callable[[float], float]


def write_comparison(path: Path, report: ValidationReport) -> Path:
    """Write the before/after report.

    Returns:
        The path written, for convenience.
    """
    findings = validation_findings(report)
    parts = [
        _header(report),
        _what_this_can_say(report),
        findings_section(findings),
        _summary_table(report),
    ]
    parts.extend(_axis_section(c) for c in report.axes.values())
    if report.notes:
        parts.append(
            "<h2>Not compared</h2><ul>"
            + "".join(f"<li>{html.escape(note)}</li>" for note in report.notes)
            + "</ul>"
        )

    document = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>RotorID validation - {html.escape(report.after.path.name)}</title>"
        f"<style>{STYLE}</style></head><body>{''.join(parts)}</body></html>"
    )
    path.write_text(document, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def _header(report: ValidationReport) -> str:
    when = report.created_utc.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"<h1>{html.escape(report.before.path.name)} &rarr; "
        f"{html.escape(report.after.path.name)}</h1>"
        f"<p class='sub'>RotorID {html.escape(report.tool_version)} &middot; {when} &middot; "
        f"{html.escape(report.after.stack)}</p>"
    )


def _what_this_can_say(report: ValidationReport) -> str:
    """The scope of the document, stated before any number in it.

    A comparison without a saved session can say the aircraft changed. It cannot
    say the tool was right, because nothing in it recorded what the tool claimed
    would happen. Leaving that distinction to be inferred from a missing column
    is how an outcome comparison gets quoted as a validation.
    """
    if report.has_predictions:
        return (
            "<div class='note'><strong>This is a validation.</strong> The predictions come "
            f"from the analysis of <code>{html.escape(report.predicted_from or '')}</code>, "
            "so the columns below compare what the tool said would happen against what the "
            "aircraft did. That is the only check in this tool that puts the model against "
            "the vehicle rather than against itself.</div>"
        )
    return (
        "<div class='note warn'><strong>This is an outcome comparison, not a validation.</strong> "
        "No saved session was supplied, so nothing here records what the tool predicted. "
        "These columns can say whether the aircraft changed and in which direction; they "
        "cannot say whether the recommendation was right. Re-run with the "
        "<code>.rotorid</code> session that produced the change to get that.</div>"
    )


def _summary_table(report: ValidationReport) -> str:
    """One row per axis: the three numbers a reader wants before the detail."""
    rows: list[tuple[str, ...]] = []
    for axis, c in report.axes.items():
        rows.append(
            (
                axis,
                _change(c.tracking_change),
                _change(c.dterm_change),
                _verdict(c),
            )
        )
    if not rows:
        return "<h2>Summary</h2><p>Neither log carries an axis the other one does.</p>"
    return "<h2>Summary</h2>" + table(
        ("Axis", "Tracking error", "D-term noise", "Prediction"),
        rows,
        numeric_from=1,
    )


def _axis_section(c: AxisComparison) -> str:
    """Everything about one axis: the numbers, then the two figures."""
    parts = [f"<h2>{html.escape(c.axis)}</h2>", _axis_table(c)]
    steps = _step_figure(c)
    if steps:
        parts.append(steps)
    spectra = _spectrum_figure(c)
    if spectra:
        parts.append(spectra)
    return "".join(parts)


def _axis_table(c: AxisComparison) -> str:
    rows: list[tuple[str, ...]] = []
    rows.append(
        (
            "Rate tracking error (RMS)",
            _number(c.before_tracking_rms, "{:.3f} rad/s"),
            _number(c.after_tracking_rms, "{:.3f} rad/s"),
        )
    )
    rows.append(
        (
            "D-term noise",
            _number(c.before_dterm_pct, "{:.2f} %"),
            _number(c.after_dterm_pct, "{:.2f} %"),
        )
    )
    if c.before_step is not None or c.after_step is not None:
        rows.append(
            (
                "Rise time (measured)",
                _step_number(c.before_step, "rise"),
                _step_number(c.after_step, "rise"),
            )
        )
        rows.append(
            (
                "Overshoot (measured)",
                _step_number(c.before_step, "overshoot"),
                _step_number(c.after_step, "overshoot"),
            )
        )
    for label, gains in (("Flown P", "kp"), ("Flown I", "ki"), ("Flown D", "kd")):
        before = getattr(c.before_gains, gains, None) if c.before_gains else None
        after = getattr(c.after_gains, gains, None) if c.after_gains else None
        if before is None and after is None:
            continue
        rows.append((label, _number(before, "{:.5f}"), _number(after, "{:.5f}")))

    body = table(("Quantity", "Before", "After"), rows, numeric_from=1)
    if c.predicted_step is None:
        return body
    predicted = table(
        ("Predicted for the recommended tune", "Value"),
        [
            ("Rise time", f"{c.predicted_step.rise_time_s * 1000:.0f} ms"),
            ("Overshoot", f"{c.predicted_step.overshoot_pct:.1f} %"),
            ("Settling time", f"{c.predicted_step.settling_time_s * 1000:.0f} ms"),
        ],
        numeric_from=1,
    )
    return body + predicted


def _step_figure(c: AxisComparison) -> str:
    """The two flown steps overlaid, with the prediction's rise time marked.

    Both measurements are drawn with their spread bands, because a mean over
    forty windows that disagree by 30% and a mean over forty that agree are not
    the same measurement, and drawing them identically would be a lie of
    presentation.
    """
    curves = [(c.before_step, "before", "var(--muted)"), (c.after_step, "after", "var(--accent)")]
    drawn = [(step, label, colour) for step, label, colour in curves if step is not None]
    if not drawn:
        return ""

    t_max = max(float(step.t[-1]) for step, _, _ in drawn)
    y_max = max(1.4, max(float(np.nanmax(step.y)) for step, _, _ in drawn) * 1.1)

    def sx(value: float) -> float:
        return _PAD + (value / t_max) * (_W - 2 * _PAD)

    def sy(value: float) -> float:
        return _H - _PAD - (value / y_max) * (_H - 2 * _PAD)

    body = [
        f"<line x1='{_PAD}' y1='{sy(1.0):.1f}' x2='{_W - _PAD}' y2='{sy(1.0):.1f}' "
        f"stroke='var(--rule)' stroke-dasharray='4 3'/>"
    ]
    for step, _label, colour in drawn:
        band = _band_path(step, sx, sy)
        if band:
            body.append(f"<path d='{band}' fill='{colour}' fill-opacity='0.15' stroke='none'/>")
        points = " ".join(
            f"{sx(float(t)):.1f},{sy(float(y)):.1f}" for t, y in zip(step.t, step.y, strict=True)
        )
        body.append(f"<polyline points='{points}' fill='none' stroke='{colour}' stroke-width='2'/>")

    legend = " &middot; ".join(
        f"<span style='color:{colour}'>&#9632;</span> {label} ({step.n_windows} windows)"
        for step, label, colour in drawn
    )
    if c.predicted_step is not None and c.predicted_step.rise_time_s <= t_max:
        x = sx(c.predicted_step.rise_time_s)
        body.append(
            f"<line x1='{x:.1f}' y1='{_PAD}' x2='{x:.1f}' y2='{_H - _PAD}' "
            f"stroke='var(--warn)' stroke-dasharray='2 3'/>"
        )
        legend += " &middot; <span style='color:var(--warn)'>&#9474;</span> predicted rise"

    return (
        f"<figure><svg viewBox='0 0 {_W} {_H}' role='img' "
        f"aria-label='measured step response before and after'>"
        f"<line x1='{_PAD}' y1='{_H - _PAD}' x2='{_W - _PAD}' y2='{_H - _PAD}' "
        f"stroke='var(--rule)'/>"
        f"<line x1='{_PAD}' y1='{_PAD}' x2='{_PAD}' y2='{_H - _PAD}' stroke='var(--rule)'/>"
        + "".join(body)
        + f"<text x='{_W - _PAD}' y='{_H - _PAD + 18}' text-anchor='end' font-size='11' "
        f"fill='var(--muted)'>{t_max:.2f} s</text>"
        f"</svg><figcaption class='rationale'>Measured step response, {legend}. "
        f"Shaded bands are the spread between the windows that were stacked."
        f"</figcaption></figure>"
    )


def _band_path(step: MeasuredStep, sx: object, sy: object) -> str:
    """A closed path tracing mean+spread out and mean-spread back."""
    if step.spread.size != step.y.size:
        return ""
    scale_x, scale_y = sx, sy
    upper = [
        f"{scale_x(float(t)):.1f},{scale_y(float(y + s)):.1f}"  # type: ignore[operator]
        for t, y, s in zip(step.t, step.y, step.spread, strict=True)
    ]
    lower = [
        f"{scale_x(float(t)):.1f},{scale_y(float(y - s)):.1f}"  # type: ignore[operator]
        for t, y, s in reversed(list(zip(step.t, step.y, step.spread, strict=True)))
    ]
    return "M" + " L".join(upper + lower) + " Z"


def _spectrum_figure(c: AxisComparison) -> str:
    """Post-filter gyro spectra: before, after, and what the design predicted.

    The prediction is drawn on the same axes as the measurement rather than in a
    panel beside it. A filter that missed its line by five hertz is obvious when
    the two curves are on top of each other and invisible when they are not.
    """
    series: list[tuple[FloatArray, FloatArray, str, str]] = []
    if c.before_noise is not None:
        series.append((c.before_noise.f_hz, c.before_noise.psd_post, "before", "var(--muted)"))
    if c.after_noise is not None:
        series.append((c.after_noise.f_hz, c.after_noise.psd_post, "after", "var(--accent)"))
    if c.predicted_psd_f_hz is not None and c.predicted_psd_post is not None:
        series.append((c.predicted_psd_f_hz, c.predicted_psd_post, "predicted", "var(--warn)"))
    if len(series) < 2:
        return ""

    f_max = min(500.0, max(float(f[-1]) for f, _, _, _ in series))
    db = [
        (f, 10.0 * np.log10(np.maximum(p, 1e-18)), label, colour) for f, p, label, colour in series
    ]
    inside = [(f <= f_max) & (f > 0.0) for f, _, _, _ in db]
    top = max(float(np.max(y[m])) for (_, y, _, _), m in zip(db, inside, strict=True) if m.any())
    bottom = top - 70.0

    def sx(value: float) -> float:
        return _PAD + float(np.log10(max(value, 1.0)) / np.log10(f_max)) * (_W - 2 * _PAD)

    def sy(value: float) -> float:
        return _H - _PAD - ((value - bottom) / (top - bottom)) * (_H - 2 * _PAD)

    body = []
    for (f, y, _, colour), mask in zip(db, inside, strict=True):
        if not mask.any():
            continue
        points = " ".join(
            f"{sx(float(fx)):.1f},{sy(float(yv)):.1f}"
            for fx, yv in zip(f[mask], np.clip(y[mask], bottom, top), strict=True)
        )
        dash = " stroke-dasharray='5 3'" if colour == "var(--warn)" else ""
        body.append(
            f"<polyline points='{points}' fill='none' stroke='{colour}' stroke-width='1.6'{dash}/>"
        )

    legend = " &middot; ".join(
        f"<span style='color:{colour}'>&#9632;</span> {label}" for _, _, label, colour in series
    )
    error = c.filter_prediction_error_db
    caption = f"Post-filter gyro spectrum, {legend}."
    if error is not None:
        caption += f" Median prediction error {error:+.1f} dB over 20&ndash;350 Hz."

    return (
        f"<figure><svg viewBox='0 0 {_W} {_H}' role='img' "
        f"aria-label='post-filter gyro spectrum before, after and predicted'>"
        f"<line x1='{_PAD}' y1='{_H - _PAD}' x2='{_W - _PAD}' y2='{_H - _PAD}' "
        f"stroke='var(--rule)'/>"
        f"<line x1='{_PAD}' y1='{_PAD}' x2='{_PAD}' y2='{_H - _PAD}' stroke='var(--rule)'/>"
        + "".join(body)
        + f"<text x='{_W - _PAD}' y='{_H - _PAD + 18}' text-anchor='end' font-size='11' "
        f"fill='var(--muted)'>{f_max:.0f} Hz</text>"
        f"</svg><figcaption class='rationale'>{caption}</figcaption></figure>"
    )


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def _number(value: float | None, template: str) -> str:
    return template.format(value) if value is not None else "not logged"


def _step_number(step: MeasuredStep | None, which: str) -> str:
    if step is None:
        return "not measurable"
    if which == "rise":
        return f"{step.metrics.rise_time_s * 1000:.0f} ms"
    return f"{step.metrics.overshoot_pct:.1f} %"


def _change(fraction: float | None) -> str:
    """A fractional change, phrased so the good direction is unmistakable."""
    if fraction is None:
        return "not comparable"
    direction = "better" if fraction < 0.0 else "worse"
    if abs(fraction) < 0.01:
        return "unchanged"
    return f"{fraction * 100:+.0f}% ({direction})"


def _verdict(c: AxisComparison) -> str:
    """One phrase for whether the prediction held, or why it was not tested."""
    if c.predicted_step is None:
        return "nothing predicted"
    if c.applied is False:
        return "gains not applied"
    holds = c.prediction_holds
    if holds is None:
        return "not measurable"
    return "confirmed" if holds else "missed"
