"""Stage 6: Design, and the live sandbox (spec sections 10.2 and 10.5).

The teaching mechanism of the whole tool. Moving the conservatism slider
re-solves the design and updates the margins, the predicted step, the phase
budget and the gains *together*, so the user learns the trade by watching four
things move at once instead of by being told a trade exists.

Two rules make that work rather than merely happen:

* **Re-solve is fast and synchronous, debounced by 50 ms.** Under the 300 ms
  budget a synchronous re-solve feels like direct manipulation; the same work
  dispatched to a thread pool arrives late, out of order, and stops feeling
  connected to the slider. Anything that cannot hold the budget belongs on the
  pool instead, and the elapsed time is displayed so a regression is visible.
* **The baseline never leaves the screen.** Every plot keeps a ghost of the
  current tune, so exploring cannot lose the reference point the user came in
  with.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from rotorid.core.analysis.margins import broken_loop, design_grid
from rotorid.core.design.controller import controller_for
from rotorid.core.design.recommend import recommend_from
from rotorid.core.types import Axis, TuneRecommendation
from rotorid.gui.state import AppState
from rotorid.gui.theme import severity_colour
from rotorid.gui.widgets.bode_plot import BodePlot
from rotorid.gui.widgets.phase_budget_plot import PhaseBudgetPlot
from rotorid.gui.widgets.step_response_plot import StepResponsePlot
from rotorid.gui.widgets.why_popover import why_button
from rotorid.gui.wizard.base import StageWidget

__all__ = ["DesignStage"]

#: Coalesce slider movement. Long enough that dragging does not queue a solve per
#: pixel, short enough that letting go feels immediate.
_DEBOUNCE_MS = 50

_MARGIN_ROWS: tuple[tuple[str, str, str], ...] = (
    ("Phase margin", "phase_margin_deg", "deg"),
    ("Gain margin", "gain_margin_db", "dB"),
    ("Crossover", "crossover_hz", "Hz"),
    ("Delay margin", "delay_margin_ms", "ms"),
    ("Sensitivity peak", "peak_sensitivity_db", "dB"),
    ("Disturbance-rejection bandwidth", "disturbance_rejection_bw_hz", "Hz"),
    ("Disturbance-rejection peak", "disturbance_rejection_peak_db", "dB"),
)

_BINDING_WORDS = {
    "phase_margin": "the phase margin target -- more gain here would leave too little "
    "tolerance for lag you have not measured",
    "gain_margin": "the gain margin target -- more gain would leave too little room for "
    "a heavier battery or worn props",
    "peak_sensitivity": "the sensitivity peak limit -- more gain would start amplifying "
    "disturbance in a narrow band rather than rejecting it",
    "identified_band": "the top of the band your log actually identified -- beyond it "
    "the model is extrapolating, and designing on an extrapolation is guessing",
    "crossover_limit_delay": "the delay in the airframe and the actuation, which no "
    "gain can compensate for",
}


class DesignStage(StageWidget):
    """Gains, margins, and the constraint that stopped them going further."""

    title = "Design"

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self._axis: Axis | None = None
        self._live: TuneRecommendation | None = None

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._resolve)

        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._controls())
        splitter.addWidget(self._plots())
        splitter.setSizes((360, 900))
        layout.addWidget(splitter, 1)

        state.analysis_finished.connect(lambda *_: self.refresh())

    # ----------------------------------------------------------------- #
    # Construction
    # ----------------------------------------------------------------- #

    def _controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self._axis_row = QHBoxLayout()
        layout.addLayout(self._axis_row)

        heading = QLabel("Conservatism")
        heading.setObjectName("Heading")
        layout.addWidget(heading)
        blurb = QLabel(
            "0 designs to the margin targets exactly. 1 holds a great deal back, "
            "trading bandwidth for tolerance to everything this log did not show: a "
            "different battery, a payload, colder air, worn props."
        )
        blurb.setObjectName("Muted")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        row = QHBoxLayout()
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 100)
        self._slider.setValue(50)
        self._slider.valueChanged.connect(self._slider_moved)
        self._slider_value = QLabel("0.50")
        row.addWidget(self._slider, 1)
        row.addWidget(self._slider_value)
        layout.addLayout(row)

        # The floor a general flight log imposes is enforced in the designer
        # whatever this slider says, so the slider has to stop where the designer
        # does. A control that moves and changes nothing is worse than one that
        # will not move: the user reads the number under their thumb and believes
        # it.
        self._floor = QLabel("")
        self._floor.setObjectName("Muted")
        self._floor.setWordWrap(True)
        self._floor.setVisible(False)
        layout.addWidget(self._floor)

        buttons = QHBoxLayout()
        reset = QPushButton("Reset to recommendation")
        reset.clicked.connect(self._reset)
        buttons.addWidget(reset)
        layout.addLayout(buttons)

        self._binding = QLabel("")
        self._binding.setWordWrap(True)
        layout.addWidget(self._binding)

        self._gains_card = QFrame()
        self._gains_card.setObjectName("Card")
        self._gains = QGridLayout(self._gains_card)
        layout.addWidget(self._gains_card)

        self._margins_card = QFrame()
        self._margins_card.setObjectName("Card")
        self._margins = QGridLayout(self._margins_card)
        layout.addWidget(self._margins_card)

        self._timing = QLabel("")
        self._timing.setObjectName("Muted")
        layout.addWidget(self._timing)
        layout.addStretch(1)
        return panel

    def _plots(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self._step = StepResponsePlot()
        self._bode = BodePlot()
        self._budget = PhaseBudgetPlot()
        layout.addWidget(self._step, 2)
        layout.addWidget(self._bode, 3)
        layout.addWidget(self._budget, 1)
        return panel

    # ----------------------------------------------------------------- #
    # Interaction
    # ----------------------------------------------------------------- #

    def _slider_moved(self, value: int) -> None:
        self._slider_value.setText(f"{value / 100:.2f}")
        self._debounce.start()

    def _reset(self) -> None:
        self._slider.setValue(round(100 * self.state.conservatism))
        self._debounce.start()

    def _resolve(self) -> None:
        """Re-solve at the current slider position, synchronously and on the clock."""
        result = self.state.result
        if result is None or self._axis is None:
            return
        analysis = result.analyses.get(self._axis)
        bundle = self.state.bundle
        if analysis is None or bundle is None:
            return

        started = time.perf_counter()
        try:
            self._live = recommend_from(
                analysis,
                bundle,
                self.state.config,
                conservatism=self._slider.value() / 100.0,
            )
        except ValueError as exc:
            self._binding.setText(str(exc))
            self._binding.setStyleSheet(f"color: {severity_colour('warning')};")
            return
        elapsed_ms = 1000 * (time.perf_counter() - started)
        self._timing.setText(
            f"re-solved in {elapsed_ms:.0f} ms"
            + ("" if elapsed_ms < 300 else " -- over the 300 ms interactive budget")
        )
        self._draw()

    # ----------------------------------------------------------------- #
    # Drawing
    # ----------------------------------------------------------------- #

    def refresh(self) -> None:
        result = self.state.result
        if result is None or not result.session.recommendations:
            return

        axes = tuple(result.session.recommendations)
        if self._axis not in axes:
            self._axis = axes[0]
        self._rebuild_axis_row(axes)
        self._apply_floor()
        self._slider.blockSignals(True)
        self._slider.setValue(round(100 * max(self.state.conservatism, self._floor_value())))
        self._slider.blockSignals(False)
        self._live = result.session.recommendations[self._axis]
        self._draw()

    def _floor_value(self) -> float:
        """Least conservatism this log's kind allows."""
        return self.state.capabilities.conservatism_floor

    def _apply_floor(self) -> None:
        """Stop the slider where the designer stops, and say why it stopped there."""
        floor = self._floor_value()
        self._slider.setMinimum(round(100 * floor))
        self._floor.setVisible(floor > 0.0)
        if floor > 0.0:
            self._floor.setText(
                f"This is a general flight log, so the design is held to at least "
                f"{floor:.2f}. A narrower identification band says less about where the "
                f"loop actually crosses over, and the margin that covers the difference "
                f"has to come from somewhere. Load a tuning flight to go below this."
            )

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

    def _draw(self) -> None:
        rec = self._live
        result = self.state.result
        bundle = self.state.bundle
        if rec is None or result is None or bundle is None or self._axis is None:
            return
        analysis = result.analyses[self._axis]

        self._draw_gains(rec)
        self._draw_margins(rec)
        self._draw_binding(rec)

        self._step.show_pair(
            rec.model,
            stack=bundle.stack,
            recommended=rec.gains,
            baseline=rec.baseline_gains,
            chain=rec.filters.chain,
            baseline_chain=rec.filters.baseline_chain,
            delay=analysis.delay,
            op=analysis.operating_point,
        )

        f_hz = design_grid(0.1, min(0.45 * bundle.loop_rate_hz, 200.0))
        for label, gains, chain, index, first in (
            ("Current", rec.baseline_gains, rec.filters.baseline_chain, 3, True),
            ("Recommended", rec.gains, rec.filters.chain, 2, False),
        ):
            loop = broken_loop(
                f_hz,
                controller_for(bundle.stack, gains, chain),
                rec.model,
                delay=analysis.delay,
                op=analysis.operating_point,
            )
            self._bode.show_loop(
                f_hz,
                loop,
                rec.margins if not first else None,
                label=label,
                index=index,
                clear=first,
            )

        self._budget.show_budget(rec.latency, rec.margins)

    def _draw_gains(self, rec: TuneRecommendation) -> None:
        _clear_grid(self._gains)
        self._gains.addWidget(_muted("Gain"), 0, 0)
        self._gains.addWidget(_muted("Current"), 0, 1)
        self._gains.addWidget(_muted("Now"), 0, 2)

        suffix = {"roll": "RLL", "pitch": "PIT", "yaw": "YAW"}[rec.axis]
        rows = (
            ("P", rec.baseline_gains.kp, rec.gains.kp, f"ATC_RAT_{suffix}_P"),
            ("I", rec.baseline_gains.ki, rec.gains.ki, f"ATC_RAT_{suffix}_I"),
            ("D", rec.baseline_gains.kd, rec.gains.kd, f"ATC_RAT_{suffix}_D"),
            ("FF", rec.baseline_gains.kff, rec.gains.kff, f"ATC_RAT_{suffix}_FF"),
        )
        for row, (name, old, new, key) in enumerate(rows, start=1):
            self._gains.addWidget(QLabel(name), row, 0)
            self._gains.addWidget(_muted(f"{old:.4g}"), row, 1)
            self._gains.addWidget(QLabel(f"{new:.4g}"), row, 2)
            button = why_button(key, rec, self)
            if button is not None:
                self._gains.addWidget(button, row, 3)

    def _draw_margins(self, rec: TuneRecommendation) -> None:
        _clear_grid(self._margins)
        for row, (label, field, units) in enumerate(_MARGIN_ROWS):
            value = float(getattr(rec.margins, field))
            self._margins.addWidget(_muted(label), row, 0)
            self._margins.addWidget(QLabel(f"{value:.2f} {units}"), row, 1)
            key = {
                "phase_margin_deg": "phase_margin",
                "gain_margin_db": "gain_margin",
                "crossover_hz": "crossover",
                "disturbance_rejection_bw_hz": "drb",
            }.get(field)
            if key is not None:
                button = why_button(key, rec, self)
                if button is not None:
                    self._margins.addWidget(button, row, 2)

        row = len(_MARGIN_ROWS)
        self._margins.addWidget(_muted("D-term noise"), row, 0)
        self._margins.addWidget(QLabel(f"{rec.dterm_noise_rms_pct:.2f} % of full output"), row, 1)
        noise_why = why_button("dterm_noise", rec, self)
        if noise_why is not None:
            self._margins.addWidget(noise_why, row, 2)

    def _draw_binding(self, rec: TuneRecommendation) -> None:
        words = _BINDING_WORDS.get(rec.binding_constraint, rec.binding_constraint.replace("_", " "))
        self._binding.setStyleSheet("")
        self._binding.setText(f"What stops the gains going higher: {words}.")


def _clear_grid(grid: QGridLayout) -> None:
    while grid.count():
        item = grid.takeAt(0)
        widget = item.widget() if item is not None else None
        if widget is not None:
            widget.deleteLater()


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Muted")
    return label
