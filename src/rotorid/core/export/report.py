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
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from rotorid.core.types import LogBundle, TuneRecommendation

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
) -> Path:
    """Write the session report.

    Args:
        recommendations: Axis name to recommendation. Axes that failed to identify
            are simply absent, and the report says which and why elsewhere.

    Returns:
        The path written, for convenience.
    """
    parts = [
        _header(bundle, config_hash, tool_version),
        _safety_block(),
    ]
    for axis, rec in recommendations.items():
        parts.append(_axis_section(axis, rec))
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


def _axis_section(axis: str, rec: TuneRecommendation) -> str:
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
        + f"<div class='note'>{html.escape(rec.filters.rationale)}</div>"
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
