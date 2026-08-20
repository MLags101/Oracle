"""Open-loop magnitude and phase, with the margins marked (spec section 10.2).

Two stacked panels sharing an x-axis: magnitude in dB and phase in degrees. The
margins are drawn on the plot rather than only printed beside it, because phase
margin *is* a distance on this picture, and a user who has seen it as a distance
once stops needing the definition.

The 0 dB line and the -180 degree line are drawn always, even when nothing
crosses them in view. They are the reference the whole reading depends on, and a
plot that hides them when convenient teaches the wrong shape.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

from rotorid.core.types import ComplexArray, FloatArray, MarginReport
from rotorid.gui.theme import Palette, palette
from rotorid.gui.widgets.plot_base import PlotCard, pen

__all__ = ["BodePlot"]

_EXPLANATION = (
    "The loop gain: what a disturbance at each frequency comes back as after "
    "going once around the controller, the filters and the airframe.\n\n"
    "Crossover is where the top panel passes 0 dB -- above it the loop stops "
    "correcting. Phase margin is how far the bottom panel is above -180 degrees "
    "at that same frequency; gain margin is how far the top panel is below 0 dB "
    "where the bottom one reaches -180.\n\n"
    "Every filter you add pulls the phase curve down. That is the trade this "
    "whole tool is about, and it is visible here as the bottom curve sagging "
    "toward the line before the top curve has finished."
)


class BodePlot(QWidget):
    """Magnitude over phase, x-linked, with the margins drawn on."""

    def __init__(self, theme: Palette | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme if theme is not None else palette("light")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.magnitude = PlotCard(
            "Open loop",
            _EXPLANATION,
            y_label="Magnitude",
            y_units="dB",
            theme=self._theme,
        )
        self.phase = PlotCard(
            "",
            _EXPLANATION,
            y_label="Phase",
            y_units="deg",
            theme=self._theme,
        )
        self.phase.plot.setXLink(self.magnitude.plot)
        layout.addWidget(self.magnitude, 2)
        layout.addWidget(self.phase, 1)

        self.magnitude.plot.addLine(y=0.0, pen=pg.mkPen(self._theme.grid, width=1))
        self.phase.plot.addLine(y=-180.0, pen=pg.mkPen(self._theme.grid, width=1))
        self._markers: list[object] = []

    def show_loop(
        self,
        f_hz: FloatArray,
        loop: ComplexArray,
        margins: MarginReport | None = None,
        *,
        label: str = "Recommended",
        index: int = 2,
        clear: bool = True,
    ) -> None:
        if clear:
            self.magnitude.clear()
            self.phase.clear()
            for marker in self._markers:
                self.magnitude.plot.removeItem(marker)
            self._markers.clear()

        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(loop), 1e-12))
        phase_deg = np.degrees(np.unwrap(np.angle(loop)))
        self.magnitude.plot.plot(f_hz, magnitude_db, pen=pen(index), name=label)
        self.phase.plot.plot(f_hz, phase_deg, pen=pen(index), name=label)

        if margins is not None and margins.crossover_hz > 0.0:
            line = pg.InfiniteLine(
                pos=float(np.log10(margins.crossover_hz)),
                angle=90,
                pen=pen(1, width=1, dashed=True),
                label=f"crossover {margins.crossover_hz:.2f} Hz",
            )
            self.magnitude.plot.addItem(line)
            self._markers.append(line)
