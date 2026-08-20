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
    QVBoxLayout,
    QWidget,
)

from rotorid.core.types import Axis, SpectralPeak
from rotorid.gui.state import AppState
from rotorid.gui.widgets.prepost_spectrum import PrePostSpectrumPlot
from rotorid.gui.wizard.base import StageWidget

__all__ = ["HealthStage"]

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

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self._axis: Axis | None = None

        layout = QVBoxLayout(self)
        self._axis_row = QHBoxLayout()
        layout.addLayout(self._axis_row)

        self._summary = QLabel(
            "Run the analysis to measure the noise. Until then this is what the log "
            "says about itself."
        )
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._spectrum = PrePostSpectrumPlot()
        layout.addWidget(self._spectrum, 3)

        self._peaks = QTableWidget(0, 5)
        self._peaks.setHorizontalHeaderLabels(
            ("Frequency", "Above floor", "Width", "What it is", "What to do about it")
        )
        self._peaks.verticalHeader().setVisible(False)
        self._peaks.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._peaks, 2)

        state.analysis_finished.connect(lambda *_: self.refresh())

    # ----------------------------------------------------------------- #

    def refresh(self) -> None:
        result = self.state.result
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
        while self._axis_row.count():
            item = self._axis_row.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
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
