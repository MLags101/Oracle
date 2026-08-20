"""Measured against predicted spectrum (spec section 10.4).

Three traces on one log-frequency axis:

* the **pre-filter** spectrum -- what the gyro actually sees,
* the **predicted post-filter** spectrum for the candidate chain, and
* the measured post-filter spectrum where the log has one.

The third is the credibility check and the reason this plot is worth more than
a parameter table. Where the *current* chain's prediction can be laid over the
measured post-filter trace, the user can see for themselves whether the filter
model in this tool describes their aircraft -- and therefore whether the
prediction for the proposed chain is worth anything.

The pre-filter trace is usually reconstructed rather than measured, because the
routinely logged gyro is post-filter on both stacks. That is stated on the plot,
not buried, because a notch deep enough to hide its own peak makes the
reconstruction blind exactly where it matters most.
"""

from __future__ import annotations

import numpy as np

from rotorid.core.types import FloatArray, NoiseProfile, SpectralPeak
from rotorid.gui.widgets.plot_base import PlotCard, pen

__all__ = ["PrePostSpectrumPlot"]

_EXPLANATION = (
    "How much vibration energy reaches the controller at each frequency.\n\n"
    "The peaks that move with the motors belong to a tracking notch. Peaks that "
    "stay put when the motors change speed are the frame resonating, and no "
    "amount of notch tracking will fix a loose arm.\n\n"
    "The predicted trace is your pre-filter spectrum pushed through the "
    "candidate filter chain. Where a measured post-filter trace is also drawn, "
    "the two should sit on top of each other for the filters you are already "
    "flying -- if they do not, the filter model does not describe your aircraft, "
    "and neither does the prediction for the proposed chain."
)

_FLOOR_DB = -140.0


class PrePostSpectrumPlot(PlotCard):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(
            "Noise spectrum",
            _EXPLANATION,
            y_label="Power",
            y_units="dB",
            **kwargs,  # type: ignore[arg-type]
        )

    def format_readout(self, x: float, y: float) -> str:
        return f"{x:.1f} Hz    {y:.0f} dB"

    def show_spectra(
        self,
        f_hz: FloatArray | None,
        *,
        pre: FloatArray | None = None,
        measured_post: FloatArray | None = None,
        predicted_post: FloatArray | None = None,
        peaks: tuple[SpectralPeak, ...] = (),
        pre_filter_source: str = "none",
    ) -> None:
        self.clear()
        if f_hz is None or f_hz.size == 0:
            return

        for values, label, index, dashed in (
            (pre, self._pre_label(pre_filter_source), 0, False),
            (measured_post, "Measured, post-filter", 1, False),
            (predicted_post, "Predicted, post-filter", 2, True),
        ):
            if values is None:
                continue
            self.plot.plot(f_hz, _db(values), pen=pen(index, dashed=dashed), name=label)

        for peak in peaks:
            self._mark(peak)

    @staticmethod
    def _pre_label(source: str) -> str:
        return {
            "measured": "Pre-filter, measured",
            "reconstructed": "Pre-filter, reconstructed from the post-filter log",
        }.get(source, "Pre-filter")

    def _mark(self, peak: SpectralPeak) -> None:
        import pyqtgraph as pg

        kind = {
            "motor_fundamental": "motor",
            "motor_harmonic": f"motor x{peak.harmonic_index or ''}",
            "structural": "frame resonance",
            "broadband": "broadband",
        }.get(peak.kind, peak.kind)
        line = pg.InfiniteLine(
            pos=float(np.log10(max(peak.f_hz, 1e-3))),
            angle=90,
            pen=pen(5 if peak.kind == "structural" else 4, width=1, dashed=True),
            label=f"{kind} {peak.f_hz:.0f} Hz",
            labelOpts={"position": 0.92, "rotateAxis": (1, 0)},
        )
        self.plot.addItem(line)


def _db(values: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore"):
        db = 10.0 * np.log10(np.maximum(np.asarray(values, dtype=np.float64), 1e-30))
    return np.asarray(np.maximum(db, _FLOOR_DB), dtype=np.float64)


def spectra_from(noise: NoiseProfile | None) -> dict[str, object]:
    """The keyword arguments :meth:`show_spectra` wants, out of a noise profile."""
    if noise is None:
        return {"f_hz": None}
    return {
        "f_hz": noise.f_hz,
        "pre": noise.psd_pre,
        "measured_post": noise.psd_post,
        "peaks": noise.peaks,
        "pre_filter_source": noise.pre_filter_source,
    }
