"""Stage 9: Validate (spec sections 5.10 and 10.2).

The last screen, and the only one that argues from flown data rather than from a
model. Every other stage builds the case; this one checks it, by putting what the
tool predicted next to what the aircraft did.

Which makes it the screen with the most to lose from vagueness. Three different
claims can be made here and they must never be confused: the aircraft changed,
the aircraft improved, and the tool was right. Only the last is a validation, and
it needs an analysis of the before-log to have been run -- otherwise nothing on
the screen recorded what was predicted, and the honest thing is to say so at the
top rather than to leave a column quietly empty.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from rotorid.core.analysis.compare import AxisComparison, ValidationReport
from rotorid.core.guidance.validation import validation_findings
from rotorid.core.types import LogBundle
from rotorid.gui.state import AppState
from rotorid.gui.theme import Palette
from rotorid.gui.widgets.findings_panel import FindingsPanel
from rotorid.gui.widgets.plot_base import PlotCard, pen
from rotorid.gui.widgets.responsive import ResponsiveRow
from rotorid.gui.wizard.base import StageWidget

__all__ = ["ValidateStage"]

_STEP_EXPLANATION = (
    "The step response deconvolved from each flight, and -- when the before-log "
    "was analysed -- the one the recommendation predicted.\n\n"
    "The two solid curves are measurements: what the aircraft actually did, "
    "stacked over every usable stick input in each log. The dashed curve is the "
    "prediction. That comparison is the only one in this tool that puts the model "
    "against the vehicle rather than against itself.\n\n"
    "The shaded bands are the spread between the windows that were stacked. A "
    "narrow band means the aircraft did the same thing every time; a wide one "
    "means the average is hiding a disagreement."
)

_SPECTRUM_EXPLANATION = (
    "Post-filter gyro noise in each flight, and what the filter design predicted "
    "for the after-flight.\n\n"
    "This is the half of a recommendation that normally goes unchecked. A gain "
    "change announces itself in how the aircraft feels; a notch that landed two "
    "hertz off the motor line does not, and the only way anybody finds out is by "
    "measuring the spectrum of a flight flown with it."
)


class ValidateStage(StageWidget):
    title = "Validate"

    def __init__(
        self,
        state: AppState,
        theme: Palette | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(state, theme, parent)
        self.setAcceptDrops(True)

        layout = self.page()
        layout.addWidget(
            self.header(
                "Validate",
                subtitle="Did it do what it said it would? Open a second flight to find out.",
            )
        )

        self._scope = QLabel()
        self._scope.setObjectName("Muted")
        self._scope.setWordWrap(True)
        layout.addWidget(self._scope)

        row = QHBoxLayout()
        self._choose = QPushButton("Choose the after-flight log...")
        self._choose.setObjectName("Primary")
        self._choose.clicked.connect(self.open_after_dialog)
        row.addWidget(self._choose)
        self._status = QLabel("No second log loaded.")
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        row.addWidget(self._status, 1)
        layout.addLayout(row)

        self._table = QTreeWidget()
        self._table.setColumnCount(4)
        self._table.setHeaderLabels(("Quantity", "Before", "After", "Predicted"))
        self._table.setRootIsDecorated(False)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(180)
        layout.addWidget(self._table)

        # Side by side when the window can hold two plots, stacked when it
        # cannot. Two plot cards in a fixed row give this page a floor wider than
        # a 1200-pixel window has left with the findings dock open.
        plots = ResponsiveRow(threshold=820)
        self._steps = PlotCard(
            "Step response, measured",
            _STEP_EXPLANATION,
            x_label="Time",
            x_units="s",
            y_label="Rate",
            y_units="rad/s",
            log_x=False,
            theme=self.theme,
        )
        self._spectra = PlotCard(
            "Post-filter gyro noise",
            _SPECTRUM_EXPLANATION,
            y_label="PSD",
            y_units="dB",
            theme=self.theme,
        )
        for plot in (self._steps, self._spectra):
            plot.setMinimumHeight(280)
        plots.add(self._steps)
        plots.add(self._spectra)
        layout.addWidget(plots, 1)

        self._findings = FindingsPanel(self.theme)
        self._findings.setMinimumHeight(200)
        layout.addWidget(self._findings, 1)

        state.after_loaded.connect(self._on_after)
        state.comparison_finished.connect(lambda _: self.refresh())
        state.comparison_failed.connect(self._on_failed)
        self.refresh()

    # ----------------------------------------------------------------- #

    def open_after_dialog(self) -> None:
        """Ask for the second log. Public so a menu action can open the same dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open the after-flight log", "", "Flight logs (*.bin *.BIN *.ulg);;All files (*)"
        )
        if path:
            self._status.setText(f"Reading {path}...")
            self.state.load_after_log(Path(path))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        """A log dropped here is the *after* log, not a replacement for the first.

        The window-level drop handler loads a new before-log and throws the
        analysis away, which is right everywhere except on this screen, where the
        user has two logs in mind and has already told us which is which by
        standing here.
        """
        urls = event.mimeData().urls()
        if urls:
            self.state.load_after_log(Path(urls[0].toLocalFile()))
            event.acceptProposedAction()

    def _on_after(self, bundle: object) -> None:
        assert isinstance(bundle, LogBundle)
        self._status.setText(f"{bundle.path.name} loaded. Comparing...")

    def _on_failed(self, message: str, _traceback: str) -> None:
        self._status.setText(message)

    # ----------------------------------------------------------------- #

    def refresh(self) -> None:
        report: ValidationReport | None = self.state.comparison
        self._scope.setText(self._scope_text(report))
        self._table.clear()
        self._steps.clear()
        self._spectra.clear()
        if report is None:
            self._findings.show_findings((), {})
            return

        self._status.setText(
            f"{report.before.path.name} vs {report.after.path.name}"
            + (f" -- {'; '.join(report.notes)}" if report.notes else "")
        )
        for comparison in report.axes.values():
            self._add_rows(comparison)
        self._table.expandAll()
        first = next(iter(report.axes.values()), None)
        if first is not None:
            self._draw_steps(first)
            self._draw_spectra(first)
        self._findings.show_findings(validation_findings(report), {})

    def _scope_text(self, report: ValidationReport | None) -> str:
        """What this screen can claim, stated before any number on it."""
        if self.state.bundle is None:
            return "Load a flight on the Load stage first; this compares it against a second one."
        if report is None:
            return (
                "Load the log from a flight flown <em>after</em> applying a recommendation. "
                "This screen compares the two, and -- if the first log has been analysed -- "
                "checks what the tool predicted against what the aircraft did."
            )
        if report.has_predictions:
            return (
                "<b>This is a validation.</b> The predictions come from the analysis of "
                f"{report.predicted_from}, so the columns below compare what the tool said "
                "would happen against what happened."
            )
        return (
            "<b>This is an outcome comparison, not a validation.</b> The first log has not "
            "been analysed, so nothing here records what was predicted. Run the analysis "
            "(Ctrl+R) to check the prediction as well as the outcome."
        )

    def _add_rows(self, c: AxisComparison) -> None:
        parent = QTreeWidgetItem((c.axis, "", "", self._verdict(c)))
        rows: list[tuple[str, str, str, str]] = [
            (
                "Rate tracking error (RMS)",
                _fmt(c.before_tracking_rms, "{:.3f} rad/s"),
                _fmt(c.after_tracking_rms, "{:.3f} rad/s"),
                "",
            ),
            (
                "D-term noise",
                _fmt(c.before_dterm_pct, "{:.2f} %"),
                _fmt(c.after_dterm_pct, "{:.2f} %"),
                "",
            ),
            (
                "Rise time",
                _step(c.before_step, "rise"),
                _step(c.after_step, "rise"),
                _fmt(
                    c.predicted_step.rise_time_s * 1000.0 if c.predicted_step else None,
                    "{:.0f} ms",
                ),
            ),
            (
                "Overshoot",
                _step(c.before_step, "overshoot"),
                _step(c.after_step, "overshoot"),
                _fmt(c.predicted_step.overshoot_pct if c.predicted_step else None, "{:.1f} %"),
            ),
        ]
        if c.filter_prediction_error_db is not None:
            rows.append(
                (
                    "Filter prediction error",
                    "",
                    f"{c.filter_prediction_error_db:+.1f} dB",
                    "0.0 dB",
                )
            )
        for row in rows:
            parent.addChild(QTreeWidgetItem(row))
        self._table.addTopLevelItem(parent)
        for column in range(4):
            self._table.resizeColumnToContents(column)

    @staticmethod
    def _verdict(c: AxisComparison) -> str:
        if c.predicted_step is None:
            return "nothing predicted"
        if c.applied is False:
            return "recommended gains were not flown"
        holds = c.prediction_holds
        if holds is None:
            return "no step to measure"
        return "prediction confirmed" if holds else "prediction MISSED"

    def _draw_steps(self, c: AxisComparison) -> None:
        """Both measurements with their spread, and the prediction as a dashed line."""
        for step, label, index in (
            (c.before_step, "Before, measured", 3),
            (c.after_step, "After, measured", 2),
        ):
            if step is None:
                continue
            self._steps.plot.plot(step.t, step.y, pen=pen(index), name=label)
            if step.spread.size == step.y.size:
                for edge in (step.y + step.spread, step.y - step.spread):
                    self._steps.plot.plot(step.t, edge, pen=pen(index, width=1, dashed=True))
        if c.predicted_step is not None and c.after_step is not None:
            self._steps.plot.addLine(
                x=c.predicted_step.rise_time_s, pen=pen(1, width=1, dashed=True)
            )
        self._steps.plot.addLine(y=1.0, pen=pen(4, width=1, dashed=True))

    def _draw_spectra(self, c: AxisComparison) -> None:
        """Before, after and predicted on one pair of axes.

        On the same axes rather than in panels side by side: a filter that missed
        its line by five hertz is obvious when the curves are on top of each other
        and invisible when they are not.
        """
        for f_hz, psd, label, index, dashed in (
            (
                c.before_noise.f_hz if c.before_noise else None,
                c.before_noise.psd_post if c.before_noise else None,
                "Before",
                3,
                False,
            ),
            (
                c.after_noise.f_hz if c.after_noise else None,
                c.after_noise.psd_post if c.after_noise else None,
                "After",
                2,
                False,
            ),
            (c.predicted_psd_f_hz, c.predicted_psd_post, "Predicted", 1, True),
        ):
            if f_hz is None or psd is None or f_hz.size != psd.size:
                continue
            usable = f_hz > 0.0
            self._spectra.plot.plot(
                f_hz[usable],
                10.0 * np.log10(np.maximum(psd[usable], 1e-18)),
                pen=pen(index, dashed=dashed),
                name=label,
            )


def _fmt(value: float | None, template: str) -> str:
    return template.format(value) if value is not None else "not logged"


def _step(step: object, which: str) -> str:
    if step is None:
        return "not measurable"
    metrics = step.metrics  # type: ignore[attr-defined]
    if which == "rise":
        return f"{metrics.rise_time_s * 1000:.0f} ms"
    return f"{metrics.overshoot_pct:.1f} %"
