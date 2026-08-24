"""Stage 3: Segment (spec section 10.2).

Which stretch of the flight the identification uses decides the band the model
is evidence over, and therefore what the recommendation is allowed to say. So
the segments are shown on the trace rather than listed as numbers: a sweep that
started before the aircraft was steady, or ended after the pilot took over, is
obvious as a picture and invisible as a table row.

The auto-detected segments are always shown, and the user may always disagree
with them.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import (
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from rotorid.core.types import ExcitationSegment, LogBundle
from rotorid.gui.state import AppState
from rotorid.gui.theme import Palette
from rotorid.gui.widgets.plot_base import PlotCard, pen
from rotorid.gui.wizard.base import StageWidget

__all__ = ["SegmentStage"]

_EXPLANATION = (
    "The flight, with the excitation the tool found shaded on it.\n\n"
    "Identification uses these stretches and nothing else. A segment that "
    "includes the moments before the sweep settled, or after the pilot took "
    "over, drags noise into the estimate and narrows the band the model can "
    "honestly claim.\n\n"
    "The commanded rate and the measured rate are drawn together: where they "
    "part company is where the loop is not keeping up, which is the same "
    "information the identification is about to extract numerically."
)


class SegmentStage(StageWidget):
    """The flight, and the parts of it the identification will use."""

    title = "Segment"

    def __init__(
        self,
        state: AppState,
        theme: Palette | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(state, theme, parent)

        layout = self.page()
        layout.addWidget(
            self.header(
                "Segment",
                subtitle=(
                    "Which stretches of the flight can support an identification, and "
                    "which are the aircraft being flown rather than being measured."
                ),
            )
        )

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._trace = PlotCard(
            "Flight",
            _EXPLANATION,
            x_label="Time",
            x_units="s",
            y_label="Rate",
            y_units="rad/s",
            log_x=False,
            theme=self.theme,
        )
        self._trace.setMinimumHeight(300)
        layout.addWidget(self._trace, 3)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(("Axis", "Kind", "Start", "End", "Confidence"))
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(150)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)

        state.log_loaded.connect(lambda *_: self.refresh())
        state.analysis_finished.connect(lambda *_: self.refresh())

    # ----------------------------------------------------------------- #

    def refresh(self) -> None:
        bundle = self.state.bundle
        if bundle is None:
            return

        segments = self._segments(bundle)
        self._draw_trace(bundle, segments)
        self._fill_table(segments)
        if segments:
            self._summary.setText(
                f"{len(segments)} usable excitation segment(s), covering "
                f"{sum(s.t_end - s.t_start for s in segments):.0f} s of the flight."
            )
        else:
            self._summary.setText(
                "No excitation found. Identification needs a sweep: fly SYSTEMID with "
                "SID_AXIS set to the rate loop you want (7 roll, 8 pitch, 9 yaw), or "
                "accept a much weaker answer from ordinary flight."
            )

    def _segments(self, bundle: LogBundle) -> tuple[ExcitationSegment, ...]:
        result = self.state.result
        if result is not None and result.session.segments:
            return tuple(result.session.segments)
        from rotorid.core.preprocess.segment import propose_segments

        return tuple(propose_segments(bundle))

    def _draw_trace(self, bundle: LogBundle, segments: tuple[ExcitationSegment, ...]) -> None:
        self._trace.clear()
        for item in list(self._trace.plot.getPlotItem().items):
            if isinstance(item, pg.LinearRegionItem):
                self._trace.plot.removeItem(item)

        axis = segments[0].axis if segments else "roll"
        for index, (key, label) in enumerate(
            (
                (f"rate.{axis}.target", "Commanded"),
                (f"rate.{axis}.measured", "Measured (post-filter)"),
            )
        ):
            signal = bundle.signals.get(key)
            if signal is None:
                continue
            self._trace.plot.plot(signal.t, signal.y, pen=pen(index), name=label)

        for segment in segments:
            region = pg.LinearRegionItem(
                values=(segment.t_start, segment.t_end),
                movable=False,
                brush=pg.mkBrush(0, 158, 115, 45),
            )
            region.setZValue(-10)
            self._trace.plot.addItem(region)

    def _fill_table(self, segments: tuple[ExcitationSegment, ...]) -> None:
        self._table.setRowCount(len(segments))
        for row, segment in enumerate(segments):
            for column, text in enumerate(
                (
                    segment.axis,
                    segment.kind.replace("_", " "),
                    f"{segment.t_start:.1f} s",
                    f"{segment.t_end:.1f} s",
                    f"{segment.confidence:.2f}",
                )
            ):
                self._table.setItem(row, column, QTableWidgetItem(text))
        self._table.resizeColumnsToContents()
