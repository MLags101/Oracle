"""Shared plot behaviour (spec section 10.3).

Every plot in the tool gets the same three things, and each of them is there for
a reason that showed up in how people read this kind of data:

* **A crosshair with a value readout.** Users of tuning tools do not read plots,
  they interrogate them -- "what is the coherence at 8 Hz?" is the actual
  question, and answering it by eye off a log axis is guesswork.
* **An "explain this plot" button.** The tool's stated purpose includes teaching,
  and a plot nobody can interpret teaches nothing. The text is per-plot and
  written where the plot is built, not centrally, so it stays true to what is
  drawn.
* **Export.** Right-click gives PNG and CSV through pyqtgraph's own menu. People
  put these in build threads and forum posts, and a screenshot of a dark-themed
  window is not that.

Frequencies are always Hz and always log-x where the interesting range spans
decades, because a linear axis from 0 to 500 Hz hides everything below 20.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rotorid.gui.theme import SERIES, Palette, palette
from rotorid.gui.widgets.log_axis import LogAxis

__all__ = ["PlotCard", "pen"]


def pen(index: int, width: int = 2, dashed: bool = False) -> pg.mkPen:
    """A series pen from the colourblind-safe palette.

    ``dashed`` is not decoration: it is the second channel that keeps two traces
    distinguishable in a monochrome print and to a reader who cannot separate the
    hues.
    """
    from PySide6.QtCore import Qt

    return pg.mkPen(
        SERIES[index % len(SERIES)],
        width=width,
        style=Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine,
    )


class PlotCard(QWidget):
    """A titled plot with a readout, a legend and an explanation."""

    def __init__(
        self,
        title: str,
        explanation: str,
        *,
        x_label: str = "Frequency",
        x_units: str = "Hz",
        y_label: str = "",
        y_units: str = "",
        log_x: bool = True,
        theme: Palette | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme if theme is not None else palette("light")
        self._explanation = explanation

        pg.setConfigOption("background", self._theme.surface)
        pg.setConfigOption("foreground", self._theme.text)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        heading = QLabel(title)
        heading.setObjectName("Heading")
        # Wrapped so the card can be narrow. An unwrapped title plus the explain
        # button gives every plot a floor of some 560 pixels, and a stage holding
        # two of them side by side then cannot fit in a 1200-pixel window with
        # the findings dock open.
        heading.setWordWrap(True)
        # The heading takes the slack, so the readout and the explain button stay
        # against the right edge without a spacer competing for the same room.
        header.addWidget(heading, 1)
        self.readout = QLabel("")
        self.readout.setObjectName("Muted")
        header.addWidget(self.readout)
        explain = QPushButton("What am I looking at?")
        explain.clicked.connect(self._explain)
        header.addWidget(explain)
        layout.addLayout(header)

        # A log axis gets ours, which spaces its labels and writes them in Hz.
        # pyqtgraph's own draws every mantissa in every decade as one tick level
        # and its anti-crowding rule exempts the first level, so the labels land
        # on top of each other. See :mod:`rotorid.gui.widgets.log_axis`.
        self.plot = pg.PlotWidget(axisItems={"bottom": LogAxis("bottom")} if log_x else None)
        self.plot.setLogMode(x=log_x, y=False)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", x_label, units=x_units or None)
        if y_label:
            self.plot.setLabel("left", y_label, units=y_units or None)
        self.plot.addLegend(offset=(-10, 10))
        layout.addWidget(self.plot, 1)

        self._log_x = log_x
        self._crosshair()

    # ----------------------------------------------------------------- #

    def _explain(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("What am I looking at?")
        box.setText(self._explanation)
        box.exec()

    def _crosshair(self) -> None:
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(self._theme.grid))
        self.plot.addItem(self._vline, ignoreBounds=True)
        self.plot.scene().sigMouseMoved.connect(self._moved)

    def _moved(self, position: object) -> None:
        view = self.plot.getPlotItem().vb
        if not self.plot.sceneBoundingRect().contains(position):
            return
        point = view.mapSceneToView(position)
        x = 10 ** point.x() if self._log_x else point.x()
        self._vline.setPos(point.x())
        self.readout.setText(self.format_readout(x, point.y()))

    def format_readout(self, x: float, y: float) -> str:
        """Override to label the axes the way this particular plot reads."""
        return f"{x:.3g} Hz    {y:.3g}"

    # ----------------------------------------------------------------- #

    def clear(self) -> None:
        """Remove the data but keep the crosshair, which is furniture, not data."""
        for item in list(self.plot.getPlotItem().listDataItems()):
            self.plot.removeItem(item)
