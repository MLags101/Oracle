"""Stage 5: Filters (spec section 10.4).

The centrepiece of the filter functionality, and the screen where the trade this
tool exists to teach is visible in one glance: the spectrum on the left says what
the filters remove, the phase budget below says what they cost, and both move
together when a control changes.

Overriding the recommendation is allowed to make things worse. That is what makes
it a sandbox rather than a set of presets, and it is the fastest way to learn
that a wider notch is not a free notch. What is *not* allowed is an unmeasured
override: every hand-built chain is put through the same joint solve as the
recommended one, so the numbers beside it are about that chain and not about the
one it replaced.
"""

from __future__ import annotations

import dataclasses

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from rotorid.core.design.filters import GYRO_LPF_LADDER
from rotorid.core.design.recommend import recommend_from
from rotorid.core.filters.chain import FilterChain
from rotorid.core.types import Axis, SpectralPeak, TuneRecommendation
from rotorid.gui.state import AppState
from rotorid.gui.widgets.param_diff_table import ParamDiffTable
from rotorid.gui.widgets.phase_budget_plot import PhaseBudgetPlot
from rotorid.gui.widgets.prepost_spectrum import PrePostSpectrumPlot
from rotorid.gui.widgets.why_popover import why_button
from rotorid.gui.wizard.base import StageWidget

__all__ = ["FiltersStage"]

_DEBOUNCE_MS = 50

#: Harmonics the firmware can notch. Beyond the third the peaks are usually below
#: the noise floor and the phase is spent for nothing, which is ArduPilot's own
#: published guidance -- but the control offers them, because an aircraft with a
#: real fourth-harmonic peak exists and the tool should not pretend otherwise.
_HARMONICS = (1, 2, 3, 4)


class FiltersStage(StageWidget):
    """What the filters remove, what they cost, and what happens if you change them."""

    title = "Filters"

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self._axis: Axis | None = None
        self._live: TuneRecommendation | None = None
        self._loading = False

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._resolve)

        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._controls())
        splitter.addWidget(self._panels())
        splitter.setSizes((380, 900))
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

        card = QFrame()
        card.setObjectName("Card")
        form = QFormLayout(card)

        self._gyro_lpf = QComboBox()
        for value in GYRO_LPF_LADDER:
            self._gyro_lpf.addItem(f"{value:.0f} Hz", float(value))
        self._gyro_lpf.currentIndexChanged.connect(self._changed)
        form.addRow("Gyro low-pass", self._gyro_lpf)

        self._dterm_lpf = QSpinBox()
        self._dterm_lpf.setRange(5, 200)
        self._dterm_lpf.setSuffix(" Hz")
        self._dterm_lpf.valueChanged.connect(self._changed)
        form.addRow("D-term low-pass", self._dterm_lpf)

        self._notch_bw = QSpinBox()
        self._notch_bw.setRange(1, 200)
        self._notch_bw.setSuffix(" Hz")
        self._notch_bw.valueChanged.connect(self._changed)
        form.addRow("Notch bandwidth", self._notch_bw)

        self._notch_att = QSpinBox()
        self._notch_att.setRange(5, 60)
        self._notch_att.setSuffix(" dB")
        self._notch_att.valueChanged.connect(self._changed)
        form.addRow("Notch attenuation", self._notch_att)

        harmonics = QHBoxLayout()
        self._harmonics: dict[int, QCheckBox] = {}
        for n in _HARMONICS:
            box = QCheckBox(f"{n}x")
            box.stateChanged.connect(self._changed)
            harmonics.addWidget(box)
            self._harmonics[n] = box
        form.addRow("Harmonics", _wrap(harmonics))

        layout.addWidget(card)

        reset = QPushButton("Reset to recommendation")
        reset.clicked.connect(self.refresh)
        layout.addWidget(reset)

        self._verdict = QLabel("")
        self._verdict.setWordWrap(True)
        layout.addWidget(self._verdict)

        self._why_row = QHBoxLayout()
        layout.addLayout(self._why_row)

        self._diff = ParamDiffTable()
        layout.addWidget(self._diff, 1)
        return panel

    def _panels(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self._spectrum = PrePostSpectrumPlot()
        self._budget = PhaseBudgetPlot()
        layout.addWidget(self._spectrum, 3)
        layout.addWidget(self._budget, 1)
        return panel

    # ----------------------------------------------------------------- #
    # Interaction
    # ----------------------------------------------------------------- #

    def _changed(self, *_: object) -> None:
        if not self._loading:
            self._debounce.start()

    def _resolve(self) -> None:
        """Re-solve against the chain the controls describe, not the recommended one."""
        result = self.state.result
        bundle = self.state.bundle
        if result is None or bundle is None or self._axis is None or self._live is None:
            return
        analysis = result.analyses.get(self._axis)
        if analysis is None:
            return

        try:
            self._live = recommend_from(
                analysis,
                bundle,
                self.state.config,
                conservatism=self.state.conservatism,
                chain_override=self._chain_from_controls(self._live.filters.chain),
            )
        except ValueError as exc:
            self._verdict.setText(f"That chain has no usable design: {exc}")
            return
        self._draw()

    def _chain_from_controls(self, base: FilterChain) -> FilterChain:
        """The chain the controls currently describe.

        Built by replacing fields on the recommended chain rather than from
        scratch, so everything the user did not touch -- the tracking source, the
        reference, the composite-notch options -- keeps the value that was
        designed for it.
        """
        notches = base.notches
        if notches:
            wanted = tuple(n for n, box in sorted(self._harmonics.items()) if box.isChecked())
            notches = (
                dataclasses.replace(
                    notches[0],
                    bandwidth_hz=float(self._notch_bw.value()),
                    attenuation_db=float(self._notch_att.value()),
                    harmonics=wanted or (1,),
                ),
                *notches[1:],
            )
        return dataclasses.replace(
            base,
            gyro_lpf_hz=float(self._gyro_lpf.currentData()),
            dterm_lpf_hz=float(self._dterm_lpf.value()),
            notches=notches,
        )

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
        self._live = result.session.recommendations[self._axis]
        self._load_controls(self._live)
        self._draw()

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

    def _load_controls(self, rec: TuneRecommendation) -> None:
        """Set the controls to the recommendation without triggering a re-solve."""
        self._loading = True
        try:
            chain = rec.filters.chain
            if chain.gyro_lpf_hz:
                index = self._gyro_lpf.findData(float(chain.gyro_lpf_hz))
                if index < 0:
                    self._gyro_lpf.addItem(f"{chain.gyro_lpf_hz:.0f} Hz", float(chain.gyro_lpf_hz))
                    index = self._gyro_lpf.count() - 1
                self._gyro_lpf.setCurrentIndex(index)
            if chain.dterm_lpf_hz:
                self._dterm_lpf.setValue(round(chain.dterm_lpf_hz))

            notch = chain.notches[0] if chain.notches else None
            enabled = notch is not None
            for widget in (self._notch_bw, self._notch_att):
                widget.setEnabled(enabled)
            for box in self._harmonics.values():
                box.setEnabled(enabled)
                box.setChecked(False)

            if notch is not None:
                self._notch_bw.setValue(round(notch.bandwidth_hz))
                self._notch_att.setValue(round(notch.attenuation_db))
                for n, box in self._harmonics.items():
                    box.setChecked(n in notch.harmonics)
        finally:
            self._loading = False

    def _draw(self) -> None:
        rec = self._live
        if rec is None or self.state.bundle is None:
            return

        self._spectrum.show_spectra(
            rec.filters.psd_f_hz,
            pre=rec.filters.psd_pre,
            predicted_post=rec.filters.predicted_psd_post,
            peaks=self._peaks(),
            pre_filter_source=self._pre_filter_source(),
        )
        self._budget.show_budget(rec.latency, rec.margins)

        if rec.filters.params:
            self._diff.show_diff(rec.filters.params, self.state.bundle.params)
        else:
            self._diff.show_nothing(
                "Nothing to change. The filters this aircraft is already flying are "
                "the ones this log supports."
            )

        self._verdict.setText(
            f"{rec.filters.chain.describe()}\n\n"
            f"Costs {rec.filters.phase_cost_deg:.1f} deg of phase at the "
            f"{rec.margins.crossover_hz:.2f} Hz crossover, out of the "
            f"{rec.margins.phase_margin_deg:.0f} deg of margin the design ended with. "
            f"D-term noise {rec.dterm_noise_rms_pct:.2f}% of full output.\n\n"
            f"{rec.filters.rationale}"
        )
        self._rebuild_why_row(rec)

    def _rebuild_why_row(self, rec: TuneRecommendation) -> None:
        while self._why_row.count():
            item = self._why_row.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        for key, label in (
            ("INS_GYRO_FILTER", "gyro low-pass"),
            ("INS_HNTCH_FREQ", "notch frequency"),
            ("INS_HNTCH_BW", "bandwidth"),
            ("INS_HNTCH_HMNCS", "harmonics"),
        ):
            button = why_button(key, rec, self)
            if button is not None:
                button.setText(f"why {label}?")
                self._why_row.addWidget(button)

    def _peaks(self) -> tuple[SpectralPeak, ...]:
        result = self.state.result
        if result is None or self._axis is None:
            return ()
        noise = result.session.noise.get(self._axis)
        return tuple(noise.peaks) if noise is not None else ()

    def _pre_filter_source(self) -> str:
        result = self.state.result
        if result is None or self._axis is None:
            return "none"
        noise = result.session.noise.get(self._axis)
        return noise.pre_filter_source if noise is not None else "none"


def _wrap(layout: QHBoxLayout) -> QWidget:
    widget = QWidget()
    widget.setLayout(layout)
    return widget
