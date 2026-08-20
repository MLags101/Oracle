"""Self-contained HTML session report (spec section 7).

Two constraints shape this module. The report has to be **self-contained** --
one file a user can attach to a forum post or keep next to the log -- so there
are no external assets and no JavaScript. And it has to be **traceable**: every
number carries the model it came from, the band it was identified over, and the
constraint that bounded it. A report that shows only the answer is worth less
than no report, because it invites the reader to trust it.

Plots are inline SVG drawn here rather than through a plotting library: the
figures are simple, and a dependency-free report is worth more than a prettier
one.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rotorid.core.guidance.explain import GLOSSARY, explain, explainable, glossary_for
from rotorid.core.types import (
    Finding,
    FlightTestPlan,
    FloatArray,
    LogBundle,
    TuneRecommendation,
)

__all__ = ["write_report"]

_STYLE = """
:root { color-scheme: light dark; --fg: #16181d; --muted: #5b6270; --bg: #ffffff;
        --rule: #d7dbe2; --accent: #1f6feb; --warn: #b8500f; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e6e8ec; --muted: #9aa3b2; --bg: #14161a; --rule: #2a2f38;
          --accent: #6ea8fe; --warn: #e2a06a; }
}
body { background: var(--bg); color: var(--fg); margin: 0 auto; max-width: 52rem;
       padding: 2rem 1.25rem 4rem;
       font: 15px/1.6 -apple-system, "Segoe UI", system-ui, sans-serif; }
h1 { font-size: 1.55rem; margin: 0 0 .25rem; }
h2 { font-size: 1.1rem; margin: 2.25rem 0 .5rem; padding-bottom: .3rem;
     border-bottom: 1px solid var(--rule); }
h3 { font-size: .95rem; margin: 1.4rem 0 .4rem; }
.sub { color: var(--muted); margin: 0 0 1.5rem; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0 1rem; font-size: .92rem; }
th, td { text-align: left; padding: .35rem .6rem; border-bottom: 1px solid var(--rule); }
th { font-weight: 600; color: var(--muted); font-weight: 500; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.note { border-left: 3px solid var(--accent); padding: .6rem .9rem; margin: 1rem 0;
        background: color-mix(in srgb, var(--accent) 7%, transparent); border-radius: 0 4px 4px 0; }
.warn { border-left-color: var(--warn);
        background: color-mix(in srgb, var(--warn) 9%, transparent); }
.rationale { color: var(--muted); font-size: .9rem; }
figure { margin: 1rem 0; overflow-x: auto; }
svg { max-width: 100%; height: auto; }
code { font: .88em ui-monospace, "SF Mono", Consolas, monospace; }
"""


def write_report(
    path: Path,
    bundle: LogBundle,
    recommendations: dict[str, TuneRecommendation],
    *,
    config_hash: str,
    tool_version: str,
    findings: tuple[Finding, ...] = (),
    plan: FlightTestPlan | None = None,
) -> Path:
    """Write the session report.

    Args:
        recommendations: Axis name to recommendation. Axes that failed to identify
            are simply absent, and the report says which and why elsewhere.
        findings: What the tool noticed. Shown before the numbers, because a
            reader who scrolls straight to the gains should have already passed
            the reason not to trust them.
        plan: The staged flight plan. Shown last, because it is what the reader
            leaves with.

    Returns:
        The path written, for convenience.
    """
    parts = [
        _header(bundle, config_hash, tool_version),
        _safety_block(),
        _findings_section(findings),
    ]
    for axis, rec in recommendations.items():
        parts.append(_axis_section(axis, rec, bundle.params))
    if plan is not None:
        parts.append(_plan_section(plan))
    parts.append(_glossary_section(recommendations))
    parts.append(_log_section(bundle))

    document = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>RotorID report - {html.escape(bundle.path.name)}</title>"
        f"<style>{_STYLE}</style></head><body>{''.join(parts)}</body></html>"
    )
    path.write_text(document, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def _header(bundle: LogBundle, config_hash: str, tool_version: str) -> str:
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"<h1>{html.escape(bundle.path.name)}</h1>"
        f"<p class='sub'>RotorID {html.escape(tool_version)} &middot; {when} &middot; "
        f"{html.escape(bundle.stack)} &middot; config <code>{html.escape(config_hash)}</code></p>"
    )


def _safety_block() -> str:
    """The warning that has to appear whether or not anyone reads it."""
    return (
        "<div class='note warn'><strong>Before you fly this.</strong> Back up your "
        "current parameters. Apply filter changes and gain changes in separate "
        "flights, or a bad outcome cannot be attributed to either. Test at altitude "
        "with room to recover. These are designed starting points with stated "
        "margins, identified at one operating point &mdash; not a validated tune. "
        "The identified model shifts with payload, battery state and prop "
        "condition.</div>"
    )


def _axis_section(axis: str, rec: TuneRecommendation, flown: dict[str, float]) -> str:
    m = rec.margins
    rows = [
        ("Phase margin", f"{m.phase_margin_deg:.1f}", "deg"),
        ("Gain margin", f"{m.gain_margin_db:.1f}", "dB"),
        ("Crossover", f"{m.crossover_hz:.2f}", "Hz"),
        ("Delay margin", f"{m.delay_margin_ms:.1f}", "ms"),
        ("Peak sensitivity", f"{m.peak_sensitivity_db:.1f}", "dB"),
        ("Disturbance-rejection bandwidth", f"{m.disturbance_rejection_bw_hz:.2f}", "Hz"),
        ("Disturbance-rejection peak", f"{m.disturbance_rejection_peak_db:.1f}", "dB"),
    ]
    gain_rows = [
        ("P", rec.baseline_gains.kp, rec.gains.kp),
        ("I", rec.baseline_gains.ki, rec.gains.ki),
        ("D", rec.baseline_gains.kd, rec.gains.kd),
        ("FF", rec.baseline_gains.kff, rec.gains.kff),
    ]

    return (
        f"<h2>{html.escape(axis.title())}</h2>"
        f"<p class='rationale'>{html.escape(rec.rationale)}</p>"
        f"<div class='note'><strong>Binding constraint:</strong> "
        f"<code>{html.escape(rec.binding_constraint)}</code> &mdash; this is what "
        f"stops the gains going higher. Confidence: {html.escape(rec.confidence)}.</div>"
        "<h3>Gains</h3>"
        + _table(
            ("Gain", "Current", "Recommended", "Change"),
            [
                (
                    name,
                    f"{old:.4g}",
                    f"{new:.4g}",
                    "n/a" if old == 0.0 else f"{new / old:.2f}x",
                )
                for name, old, new in gain_rows
            ],
            numeric_from=1,
        )
        + "<h3>Identified airframe</h3>"
        + _model_table(rec)
        + "<h3>Achieved margins</h3>"
        + _table(
            ("Metric", "Value", "Unit"),
            [(label, value, unit) for label, value, unit in rows],
            numeric_from=1,
            numeric_to=2,
        )
        + "<h3>Where the phase goes at crossover</h3>"
        + _budget_figure(rec)
        + "<h3>Predicted step</h3>"
        + _step_table(rec)
        + "<h3>Filters</h3>"
        + _filter_section(rec, flown)
        + "<h3>Why these numbers</h3>"
        + _why_section(rec)
    )


def _model_table(rec: TuneRecommendation) -> str:
    model = rec.model
    rows = [(name, f"{value:.4g}") for name, value in sorted(model.params.items())]
    rows += [
        ("structure", model.structure),
        ("valid band", f"{model.valid_band_hz[0]:.2f} - {model.valid_band_hz[1]:.1f} Hz"),
        ("mean coherence", f"{model.coherence_mean:.3f}"),
        ("fit residual", f"{model.fit_rms_db:.2f} dB / {model.fit_rms_deg:.1f} deg"),
        ("filters removed by", model.filter_deconvolution),
    ]
    return _table(("Quantity", "Value"), rows, numeric_from=1)


def _step_table(rec: TuneRecommendation) -> str:
    s = rec.predicted_step
    return _table(
        ("Metric", "Value"),
        [
            ("Rise time (10-90%)", f"{s.rise_time_s * 1000.0:.0f} ms"),
            ("Overshoot", f"{s.overshoot_pct:.1f} %"),
            ("Settling time (2%)", f"{s.settling_time_s * 1000.0:.0f} ms"),
            ("Peak time", f"{s.peak_time_s * 1000.0:.0f} ms"),
        ],
        numeric_from=1,
    )


def _log_section(bundle: LogBundle) -> str:
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in bundle.warnings)
    warning_block = (
        f"<div class='note warn'><strong>Reading this log:</strong><ul>{warnings}</ul></div>"
        if warnings
        else ""
    )
    return (
        "<h2>Log</h2>"
        + _table(
            ("Property", "Value"),
            [
                ("File", bundle.path.name),
                ("Firmware", bundle.firmware_version or "not recorded"),
                ("Analysis grid", f"{bundle.sample_rate_hz:.0f} Hz"),
                ("Loop rate", f"{bundle.loop_rate_hz:.0f} Hz"),
                ("Gyro rate", f"{bundle.gyro_sample_rate_hz:.0f} Hz"),
                ("Signals extracted", str(len(bundle.signals))),
                ("Parameters", str(len(bundle.params))),
            ],
            numeric_from=1,
        )
        + warning_block
    )


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    numeric_from: int = 99,
    numeric_to: int = 99,
) -> str:
    def cls(i: int) -> str:
        return " class='num'" if numeric_from <= i <= numeric_to else ""

    head = "".join(f"<th{cls(i)}>{html.escape(h)}</th>" for i, h in enumerate(headers))
    body = "".join(
        "<tr>"
        + "".join(f"<td{cls(i)}>{html.escape(str(c))}</td>" for i, c in enumerate(row))
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _budget_figure(rec: TuneRecommendation) -> str:
    """Stacked bar of the phase-lag contributions at the design crossover.

    The single most explanatory picture the tool draws: it shows at a glance
    whether the vehicle is limited by its filters, by its ESC, or by the airframe
    itself, and therefore whether tuning gains can help at all.
    """
    items = [(label, value) for label, value in rec.latency.items() if abs(value) > 0.05]
    if not items:
        return "<p class='rationale'>No measurable phase lag at crossover.</p>"

    total = sum(value for _, value in items)
    width, height, left = 640.0, 26.0 * len(items) + 30.0, 150.0
    scale = (width - left - 60.0) / max(total, 1e-9)

    bars = []
    for i, (label, value) in enumerate(items):
        y = 8.0 + 26.0 * i
        bar = max(1.0, value * scale)
        bars.append(
            f"<text x='{left - 8:.0f}' y='{y + 13:.0f}' text-anchor='end' "
            f"font-size='12' fill='currentColor'>{html.escape(label)}</text>"
            f"<rect x='{left:.0f}' y='{y:.0f}' width='{bar:.1f}' height='18' rx='2' "
            f"fill='currentColor' opacity='{0.35 + 0.5 * value / max(total, 1e-9):.2f}'/>"
            f"<text x='{left + bar + 6:.1f}' y='{y + 13:.0f}' font-size='12' "
            f"fill='currentColor' opacity='.7'>{value:.1f}&deg;</text>"
        )

    return (
        f"<figure><svg viewBox='0 0 {width:.0f} {height:.0f}' role='img' "
        f"aria-label='Phase lag contributions at {rec.latency.at_hz:.2f} Hz'>"
        + "".join(bars)
        + "</svg><figcaption class='rationale'>Contributions at the "
        f"{rec.latency.at_hz:.2f} Hz crossover. The D-term filter sits in the "
        "derivative branch only, so it is shown here but does not add to the "
        "common-path total of "
        f"{rec.latency.common_path_deg:.1f}&deg;.</figcaption></figure>"
    )


def _filter_section(rec: TuneRecommendation, flown: dict[str, float]) -> str:
    """The filter half of the recommendation: what changes, what it costs, and why.

    Filter parameters are shown as a diff against what was flown rather than as a
    list of values, because the value that matters to the reader is the one that
    is different.
    """
    filters = rec.filters
    parts = [f"<p class='rationale'>{html.escape(filters.rationale)}</p>"]

    if filters.params:
        parts.append(
            _table(
                ("Parameter", "Current", "Recommended"),
                [
                    (
                        name,
                        _format_param(flown.get(name)),
                        f"{value:g}",
                    )
                    for name, value in sorted(filters.params.items())
                ],
                numeric_from=1,
                numeric_to=2,
            )
        )
        parts.append(
            "<div class='note warn'><strong>Fly these on their own.</strong> Apply the "
            "filter changes, fly, and confirm the vehicle is still controllable before "
            "applying the gains. Changing both at once makes a bad outcome impossible "
            "to attribute.</div>"
        )
    else:
        parts.append(
            "<p class='rationale'>No filter parameter changes are proposed, so nothing "
            "here needs to be written to the vehicle.</p>"
        )

    parts.append(
        "<p class='rationale'>Chain: <code>"
        + html.escape(filters.chain.describe())
        + f"</code> &middot; {filters.phase_cost_deg:.1f}&deg; of phase at crossover "
        + (
            f"&middot; D-term output noise {rec.dterm_noise_rms_pct:.1f}% of full scale."
            if math.isfinite(rec.dterm_noise_rms_pct)
            else "&middot; D-term output noise was not measured: this log carries no "
            "usable noise spectrum."
        )
        + "</p>"
    )
    parts.append(_spectrum_figure(rec))
    if filters.rejected:
        rows = "".join(
            f"<tr><td>{html.escape(alternative)}</td><td>{html.escape(why)}</td></tr>"
            for alternative, why in filters.rejected
        )
        parts.append(
            "<details><summary class='rationale'>Alternatives considered and why they "
            "lost</summary><table><tbody>" + rows + "</tbody></table></details>"
        )
    return "".join(parts)


def _format_param(value: float | None) -> str:
    """A parameter value as flown, or a note that the log never recorded it."""
    return "not logged" if value is None else f"{value:g}"


def _spectrum_figure(rec: TuneRecommendation) -> str:
    """Measured pre-filter gyro spectrum against the one the new chain would leave.

    The picture that makes the filter recommendation arguable: the reader can see
    which peaks were removed, which were left, and how far down the floor went.
    """
    filters = rec.filters
    f = filters.psd_f_hz
    pre = filters.psd_pre
    post = filters.predicted_psd_post
    if f is None or pre is None or post is None:
        return "<p class='rationale'>No noise spectrum was measured for this axis.</p>"

    band = (f >= 5.0) & (f <= min(float(f[-1]), 500.0)) & (pre > 0.0) & (post > 0.0)
    if band.sum() < 8:
        return "<p class='rationale'>The noise spectrum is too narrow to plot.</p>"

    f_band = f[band]
    pre_db = 10.0 * np.log10(pre[band])
    post_db = 10.0 * np.log10(post[band])

    width, height, pad = 640.0, 240.0, 34.0
    x0, x1 = float(np.log10(f_band[0])), float(np.log10(f_band[-1]))
    y1 = float(max(pre_db.max(), post_db.max()))
    y0 = float(min(pre_db.min(), post_db.min(), y1 - 20.0))

    def sx(value: float) -> float:
        return float(pad + (np.log10(value) - x0) / max(x1 - x0, 1e-9) * (width - 2.0 * pad))

    def sy(value: float) -> float:
        return height - pad - (value - y0) / max(y1 - y0, 1e-9) * (height - 2.0 * pad)

    def path(values: FloatArray) -> str:
        return " ".join(
            f"{sx(float(freq)):.1f},{sy(float(db)):.1f}"
            for freq, db in zip(f_band, values, strict=True)
        )

    ticks = "".join(
        f"<text x='{sx(decade):.1f}' y='{height - pad + 14:.0f}' font-size='11' "
        f"text-anchor='middle' fill='currentColor' opacity='.6'>{decade:g}</text>"
        for decade in (10.0, 100.0)
        if f_band[0] <= decade <= f_band[-1]
    )

    return (
        f"<figure><svg viewBox='0 0 {width:.0f} {height:.0f}' role='img' "
        f"aria-label='Gyro noise spectrum before and after the recommended filters'>"
        f"<polyline points='{path(pre_db)}' fill='none' stroke='currentColor' "
        f"stroke-width='1' opacity='.35'/>"
        f"<polyline points='{path(post_db)}' fill='none' stroke='currentColor' "
        f"stroke-width='1.5'/>"
        f"{ticks}"
        f"<text x='{pad:.0f}' y='{pad - 12:.0f}' font-size='11' fill='currentColor' "
        f"opacity='.7'>dB, gyro noise vs Hz &mdash; faint: unfiltered, solid: with the "
        f"recommended chain</text>"
        "</svg><figcaption class='rationale'>Measured pre-filter spectrum against the "
        "one the recommended chain would leave. Peaks still standing above the floor "
        "here are the ones no filter was worth spending phase on.</figcaption></figure>"
    )


_SEVERITY_LABEL = {
    "blocker": ("Blocking", "warn"),
    "warning": ("Warning", "warn"),
    "info": ("Note", ""),
    "good": ("Good", ""),
}


def _findings_section(findings: tuple[Finding, ...]) -> str:
    """What the tool noticed, worst first, each with the evidence behind it."""
    if not findings:
        return ""

    blocks = []
    for finding in findings:
        label, style = _SEVERITY_LABEL[finding.severity]
        evidence = (
            "<p class='rationale'>"
            + " &middot; ".join(
                f"{html.escape(k)} = {v:.4g}" for k, v in sorted(finding.evidence.items())
            )
            + "</p>"
            if finding.evidence
            else ""
        )
        blocks.append(
            f"<div class='note {style}'><strong>{label}: "
            f"{html.escape(finding.title)}</strong> "
            f"<code>{html.escape(finding.code)}</code>"
            f"<p>{html.escape(finding.detail)}</p>"
            f"<p><strong>What to do:</strong> {html.escape(finding.action)}</p>"
            f"{evidence}</div>"
        )

    blockers = sum(1 for f in findings if f.severity == "blocker")
    heading = "<h2>What the tool noticed</h2>"
    if blockers:
        heading += (
            f"<p class='sub'>{blockers} blocking finding(s). Exports stay disabled until "
            f"each one is acknowledged, because acting on this analysis means accepting "
            f"a stated risk rather than overlooking it.</p>"
        )
    return heading + "".join(blocks)


def _plan_section(plan: FlightTestPlan) -> str:
    """The ordered flights. One change at a time, each with its own check."""
    if not plan.stages:
        return ""

    blocks = [f"<h2>Next flights</h2><div class='note'>{html.escape(plan.preamble)}</div>"]
    for stage in plan.stages:
        changes = _table(
            ("Parameter", "Set to"),
            [(name, f"{value:g}") for name, value in sorted(stage.changes.items())],
            numeric_from=1,
        )
        watch = "".join(f"<li>{html.escape(item)}</li>" for item in stage.watch_in_flight)
        check = "".join(f"<li>{html.escape(item)}</li>" for item in stage.check_in_log)
        because = (
            "<p class='rationale'>Prompted by: "
            + ", ".join(f"<code>{html.escape(c)}</code>" for c in stage.motivating_findings)
            + "</p>"
            if stage.motivating_findings
            else ""
        )
        blocks.append(
            f"<h3>Flight {stage.index}: {html.escape(stage.title)}</h3>"
            f"{because}{changes}"
            f"<p class='rationale'><strong>Watch for:</strong></p>"
            f"<ul class='rationale'>{watch}</ul>"
            f"<p class='rationale'><strong>Then check in the log:</strong></p>"
            f"<ul class='rationale'>{check}</ul>"
        )
    return "".join(blocks)


def _why_section(rec: TuneRecommendation) -> str:
    """The trace behind each recommended number, one disclosure per number.

    Collapsed rather than inline: a reader who trusts the tool should not have to
    scroll past the reasoning, and a reader who does not should not have to ask
    for it.
    """
    blocks = []
    for key in explainable(rec):
        exp = explain(key, rec)
        if exp is None:  # pragma: no cover - explainable() only offers live keys
            continue
        reasons = "".join(f"<li>{html.escape(line)}</li>" for line in exp.because)
        terms = (
            "<p class='rationale'>See: "
            + ", ".join(html.escape(e.term) for e in glossary_for(exp))
            + "</p>"
            if exp.glossary
            else ""
        )
        blocks.append(
            f"<details><summary><strong>{html.escape(exp.title)}</strong> = "
            f"{html.escape(exp.value)}</summary>"
            f"<p>{html.escape(exp.headline)}</p>"
            f"<ul class='rationale'>{reasons}</ul>{terms}</details>"
        )
    return "".join(blocks)


def _glossary_section(recommendations: dict[str, TuneRecommendation]) -> str:
    """Definitions for every term the report actually used, and no others."""
    used: list[str] = []
    for rec in recommendations.values():
        for key in explainable(rec):
            exp = explain(key, rec)
            if exp is None:  # pragma: no cover
                continue
            used += [term for term in exp.glossary if term not in used]
    if not used:
        return ""

    entries = "".join(
        f"<details><summary><strong>{html.escape(GLOSSARY[term].term)}</strong> &mdash; "
        f"{html.escape(GLOSSARY[term].short)}</summary>"
        f"<p class='rationale'>{html.escape(GLOSSARY[term].detail)}</p></details>"
        for term in used
        if term in GLOSSARY
    )
    return "<h2>Glossary</h2>" + entries
