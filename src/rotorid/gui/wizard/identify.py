"""Stage 4: Identify (spec section 10.2).

Three curves and the argument between them:

* what was **measured** from the log -- the airframe with the flown filters
  already in it, because both firmware stacks log the filtered gyro,
* the **modelled filter chain** built from the logged parameters, and
* the **airframe** left when that chain is divided back out, with the parametric
  fit drawn over it.

The measured trace should show notch dips where the modelled chain predicts
them. When it does, the filter model describes this aircraft and everything
downstream is standing on something. When it does not, the notch never tracked,
or the parameters in the log are not the ones that were flying, and the honest
thing is to see that here rather than to discover it as a gain that oscillates.

Coherence sits underneath with the valid band shaded, because the model is only
evidence where the input explains the output.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from rotorid.core.analysis.model_eval import airframe_response
from rotorid.core.design.recommend import AxisAnalysis
from rotorid.core.types import Axis
from rotorid.gui.state import AppState
from rotorid.gui.theme import Palette
from rotorid.gui.widgets.layouts import clear
from rotorid.gui.widgets.plot_base import PlotCard, pen
from rotorid.gui.widgets.responsive import FlowLayout
from rotorid.gui.wizard.base import StageWidget

__all__ = ["IdentifyStage"]

_MAGNITUDE_EXPLANATION = (
    "How strongly the aircraft responds to a rate command at each frequency.\n\n"
    "The measured trace has your filters in it, because the gyro your flight "
    "controller logs has already been filtered. The airframe trace is that same "
    "measurement with the modelled filter chain divided back out -- and it is "
    "the airframe, not the measurement, that the gains are designed against.\n\n"
    "Keeping those two apart is the single thing this tool cannot get wrong. "
    "Count the filters twice and every recommended gain comes out timid; miss "
    "them and the recommendation oscillates."
)

_COHERENCE_EXPLANATION = (
    "How much of the measured response the commanded input actually explains, "
    "from 0 to 1, at each frequency.\n\n"
    "Low coherence means something other than your input is moving the aircraft "
    "-- wind, noise, or a nonlinearity -- and the model over that band is not "
    "evidence about anything. The shaded region is where it passed the gate; the "
    "airframe is fitted there and nowhere else.\n\n"
    "A narrow shaded band is the usual reason a recommendation comes back with "
    "low confidence. The fix is a better sweep, not a different setting."
)


class IdentifyStage(StageWidget):
    """What was measured, what was modelled, and what is left of the aircraft."""

    title = "Identify"

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
                "Identify",
                subtitle=(
                    "The airframe recovered from those segments, and how far the "
                    "measurement can be trusted at each frequency."
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

        self._verdict = QLabel("")
        self._verdict.setWordWrap(True)
        layout.addWidget(self._verdict)

        self._magnitude = PlotCard(
            "Response",
            _MAGNITUDE_EXPLANATION,
            y_label="Magnitude",
            y_units="dB",
            theme=self.theme,
        )
        self._coherence = PlotCard(
            "Coherence",
            _COHERENCE_EXPLANATION,
            y_label="Coherence",
            y_units="",
            theme=self.theme,
        )
        self._magnitude.setMinimumHeight(280)
        self._coherence.setMinimumHeight(180)
        self._coherence.plot.setXLink(self._magnitude.plot)
        self._coherence.plot.setYRange(0.0, 1.05)
        layout.addWidget(self._magnitude, 3)
        layout.addWidget(self._coherence, 1)

        state.analysis_finished.connect(lambda *_: self.refresh())

    # ----------------------------------------------------------------- #

    def refresh(self) -> None:
        result = self.state.result
        if result is None or not result.analyses:
            return
        axes = tuple(result.analyses)
        if self._axis not in axes:
            self._axis = axes[0]
        self._rebuild_axis_row(axes)
        self._draw()

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

    def _draw(self) -> None:
        result = self.state.result
        if result is None or self._axis is None:
            return
        analysis = result.analyses[self._axis]
        frf = analysis.effective.frf
        f_hz = frf.f_hz
        op = analysis.operating_point

        self._magnitude.clear()
        self._coherence.clear()

        self._magnitude.plot.plot(f_hz, _db(frf.H), pen=pen(0), name="Measured (filters included)")
        self._magnitude.plot.plot(
            f_hz,
            _db(analysis.chain.sensor_response(f_hz, op)),
            pen=pen(1),
            name="Modelled filter chain",
        )
        self._magnitude.plot.plot(
            f_hz,
            _db(airframe_response(analysis.airframe, f_hz)),
            pen=pen(2, dashed=True),
            name="Airframe, fitted",
        )
        self._coherence.plot.plot(f_hz, frf.coherence, pen=pen(0), name="Coherence")
        self._shade_valid_band(analysis.airframe.valid_band_hz)

        model = analysis.airframe
        params = "  ".join(f"{k} = {v:.4g}" for k, v in sorted(model.params.items()))
        self._verdict.setText(
            f"{model.structure}: {params}\n"
            f"Fitted over {model.valid_band_hz[0]:.2f}-{model.valid_band_hz[1]:.1f} Hz at "
            f"mean coherence {model.coherence_mean:.2f}, residual "
            f"{model.fit_rms_db:.2f} dB / {model.fit_rms_deg:.1f} deg. "
            f"Filters removed by the {model.filter_deconvolution} route, from "
            f"{len(analysis.segments)} segment(s). " + _loop_removal(analysis)
        )

    def _shade_valid_band(self, band: tuple[float, float]) -> None:
        if band[0] <= 0.0 or band[1] <= band[0]:
            return
        region = pg.LinearRegionItem(
            values=(float(np.log10(band[0])), float(np.log10(band[1]))),
            movable=False,
            brush=pg.mkBrush(0, 114, 178, 40),
        )
        region.setZValue(-10)
        self._coherence.plot.addItem(region)


def _db(values: object) -> object:
    array = np.abs(np.asarray(values))
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.maximum(array, 1e-12))


def _loop_removal(analysis: AxisAnalysis) -> str:
    """One sentence on how the feedback loop was divided out of the measurement.

    The screen already says how the *filters* were removed. This is the other
    half, and the more consequential one: an estimate taken from the mixer
    command alone under feedback describes the controller as much as the
    aircraft, and a user looking at a Bode plot has no way to tell from the curve
    which of the two they are looking at.
    """
    plant = analysis.effective
    if not plant.unbiased:
        return (
            "The loop could not be divided out -- nothing in this log is independent "
            "of the gyro -- so this curve is biased by the controller's own reaction "
            "to noise."
        )
    return (
        f"Loop divided out against {plant.instrument}; reading it straight from the "
        f"mixer command instead would move it {plant.bias_db:+.1f} dB and "
        f"{plant.bias_deg:+.1f} deg."
    )
