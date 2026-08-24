"""Stage 2: Health and Noise (spec section 10.2).

This comes *before* identification, and the order is the point. A model
identified from a log with mistracking notches, saturated motors or a clipping
gyro is not a weak model, it is confident nonsense -- it will fit beautifully and
describe something that is not the aircraft. Looking at the noise first is what
stops the rest of the session being built on that.

The peak inventory is the part worth reading closely. A peak that moves with the
motors wants a tracking notch. A peak that stays where it is however fast the
motors turn is the airframe resonating, and no notch setting will fix a loose
arm or a cracked mount -- the tool says so rather than quietly notching it,
because a notch there buys phase lag and hides the evidence.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from rotorid.core.types import Axis, Finding, SpectralPeak
from rotorid.gui.state import AppState
from rotorid.gui.theme import Palette
from rotorid.gui.widgets.layouts import clear
from rotorid.gui.widgets.prepost_spectrum import PrePostSpectrumPlot
from rotorid.gui.widgets.responsive import FlowLayout
from rotorid.gui.wizard.base import StageWidget

__all__ = ["HealthStage"]

#: The findings that gate everything below them on this page, worst first. Read
#: from the session rather than recomputed: a stage that re-ran the check could
#: show a different verdict than the report, which is the one thing a health
#: screen must never do.
_GATING_CODES = (
    "OSCILLATION_DETECTED",
    "ACCEL_CLIPPING",
    "VIBRATION_HIGH",
    "VIBRATION_LOW",
    "VIBRATION_NOT_LOGGED",
)

_KINDS = {
    "motor_fundamental": (
        "Motor",
        "Turns with the motors. A tracking notch follows it; a fixed one will not.",
    ),
    "motor_harmonic": (
        "Motor harmonic",
        "A multiple of motor speed. Worth notching only if it is large enough to pay "
        "for the phase the extra notch costs.",
    ),
    "structural": (
        "Frame resonance",
        "Stays put however fast the motors turn, so it is the airframe ringing. Fix it "
        "mechanically -- a notch here costs phase and hides the evidence.",
    ),
    "broadband": (
        "Broadband",
        "No single frequency to notch. This is a low-pass or a mechanical problem.",
    ),
}


class HealthStage(StageWidget):
    """What the gyro is actually seeing, before anything is identified from it."""

    title = "Health & Noise"

    def __init__(
        self,
        state: AppState,
        theme: Palette | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(state, theme, parent)
        self._axis: Axis | None = None

        layout = self.page()
        layout.addWidget(
            self.header(
                "Health & Noise",
                subtitle=(
                    "What the gyro is hearing, before anything is identified from it. "
                    "A model fitted to a shaking frame fits beautifully and describes "
                    "something that is not the aircraft."
                ),
            )
        )

        self._axis_row = FlowLayout(spacing=6)
        axis_row = QHBoxLayout()
        axis_label = QLabel("Axis")
        axis_label.setObjectName("Eyebrow")
        axis_row.addWidget(axis_label)
        axis_row.addLayout(self._axis_row)
        axis_row.addStretch(1)
        layout.addLayout(axis_row)

        # Above the spectrum, deliberately. If the frame is shaking or the
        # accelerometers are clipping, nothing further down this page is a
        # measurement of the aircraft, and the layout should say so before the
        # user starts reading peaks.
        self._gate = QLabel()
        self._gate.setWordWrap(True)
        self._gate.setVisible(False)
        layout.addWidget(self._gate)

        self._summary = QLabel(
            "Run the analysis to measure the noise. Until then this is what the log "
            "says about itself."
        )
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._spectrum = PrePostSpectrumPlot(theme=self.theme)
        self._spectrum.setMinimumHeight(300)
        layout.addWidget(self._spectrum, 3)

        self._peaks = QTableWidget(0, 5)
        self._peaks.setHorizontalHeaderLabels(
            ("Frequency", "Above floor", "Width", "What it is", "What to do about it")
        )
        self._peaks.verticalHeader().setVisible(False)
        self._peaks.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._peaks.setAlternatingRowColors(True)
        self._peaks.setWordWrap(True)
        self._peaks.setMinimumHeight(160)
        self._peaks.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._peaks, 2)

        state.analysis_finished.connect(lambda *_: self.refresh())

    # ----------------------------------------------------------------- #

    def refresh(self) -> None:
        result = self.state.result
        self._show_gate(result.session.findings if result is not None else ())
        if result is None or not result.session.noise:
            self._peaks.setRowCount(0)
            self._spectrum.show_spectra(None)
            return

        axes = tuple(result.session.noise)
        if self._axis not in axes:
            self._axis = axes[0]
        self._rebuild_axis_row(axes)

        noise = result.session.noise[self._axis]
        self._spectrum.show_spectra(
            noise.f_hz,
            pre=noise.psd_pre,
            measured_post=noise.psd_post,
            peaks=noise.peaks,
            pre_filter_source=noise.pre_filter_source,
        )
        self._fill_peaks(noise.peaks)
        self._summary.setText(
            f"Noise floor {noise.noise_floor_db:.0f} dB, {len(noise.peaks)} peak(s) above it. "
            f"Pre-filter spectrum: {self._source_words(noise.pre_filter_source)}"
        )

    def _show_gate(self, findings: tuple[Finding, ...]) -> None:
        """The vibration verdict, or nothing if the analysis has not run."""
        by_code = {f.code: f for f in findings}
        found = [by_code[code] for code in _GATING_CODES if code in by_code]
        if not found:
            self._gate.setVisible(False)
            return
        worst = found[0]
        self.banner(self._gate, worst.severity)
        self._gate.setText("\n\n".join(f"{f.title}\n{f.detail} {f.action}".strip() for f in found))
        self._gate.setVisible(True)

    @staticmethod
    def _source_words(source: str) -> str:
        return {
            "measured": "measured directly, from batch-sampled pre-filter gyro.",
            "reconstructed": (
                "reconstructed by dividing the flown filter chain back out of the "
                "logged post-filter gyro. Where a notch is deep, this reconstruction "
                "is blind -- the peak it removed cannot be recovered from the quiet "
                "it produced, so a working notch is never removed on that evidence."
            ),
        }.get(source, "not available in this log.")

    def _rebuild_axis_row(self, axes: tuple[Axis, ...]) -> None:
        clear(self._axis_row)
        for axis in axes:
            button = QPushButton(axis.title())
            button.setCheckable(True)
            button.setChecked(axis == self._axis)
            button.clicked.connect(lambda _checked=False, a=axis: self._pick_axis(a))
            self._axis_row.addWidget(button)

    def _pick_axis(self, axis: Axis) -> None:
        self._axis = axis
        self.refresh()

    def _fill_peaks(self, peaks: tuple[SpectralPeak, ...]) -> None:
        self._peaks.setRowCount(len(peaks))
        for row, peak in enumerate(peaks):
            kind, advice = _KINDS.get(peak.kind, (peak.kind, ""))
            if peak.kind == "motor_harmonic" and peak.harmonic_index:
                kind = f"Motor harmonic {peak.harmonic_index}x"
            for column, text in enumerate(
                (
                    f"{peak.f_hz:.0f} Hz",
                    f"{peak.magnitude_db:.0f} dB",
                    f"{peak.width_hz:.0f} Hz",
                    kind,
                    advice,
                )
            ):
                self._peaks.setItem(row, column, QTableWidgetItem(text))
        self._peaks.resizeColumnsToContents()
