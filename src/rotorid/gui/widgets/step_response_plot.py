"""Predicted step response, recommended against current (spec section 10.2).

The step is the plot users trust, because it is the only one that looks like
flying. It is also the one most easily over-read: it is a *prediction* from an
identified model, at one operating point, and it is drawn against the current
gains' prediction rather than against anything measured. Both traces come from
the same model, so the comparison between them is much better evidence than
either curve on its own -- which is exactly what the explanation says.
"""

from __future__ import annotations

import numpy as np

from rotorid.core.analysis.step import step_response
from rotorid.core.design.controller import controller_for
from rotorid.core.types import AirframeModel, GainSet
from rotorid.gui.widgets.plot_base import PlotCard, pen

__all__ = ["StepResponsePlot"]

_EXPLANATION = (
    "The rate the vehicle would reach if you asked for a step change of 1 rad/s, "
    "predicted from the identified model.\n\n"
    "Both traces are predictions from the same model -- one with your current "
    "gains, one with the recommended ones -- so the difference between them is "
    "far better evidence than either curve alone. Neither is a measurement.\n\n"
    "Read the first rise for quickness, the first peak for overshoot, and the "
    "settling for damping. A prediction that rings here will ring in the air."
)


class StepResponsePlot(PlotCard):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(
            "Predicted step response",
            _EXPLANATION,
            x_label="Time",
            x_units="s",
            y_label="Rate",
            y_units="rad/s",
            log_x=False,
            **kwargs,  # type: ignore[arg-type]
        )

    def format_readout(self, x: float, y: float) -> str:
        return f"{x * 1000:.0f} ms    {y:.3f} rad/s"

    def show_pair(
        self,
        airframe: AirframeModel,
        *,
        stack: str,
        recommended: GainSet,
        baseline: GainSet,
        chain: object,
        baseline_chain: object,
        delay: object,
        op: object = None,
        duration_s: float = 1.5,
    ) -> None:
        """Draw the recommended response over a ghost of the current one."""
        self.clear()
        for label, gains, this_chain, index, dashed in (
            ("Current", baseline, baseline_chain, 3, True),
            ("Recommended", recommended, chain, 2, False),
        ):
            t, y = step_response(
                controller_for(stack, gains, this_chain),  # type: ignore[arg-type]
                airframe,
                delay=delay,  # type: ignore[arg-type]
                op=op,  # type: ignore[arg-type]
                duration_s=duration_s,
            )
            self.plot.plot(t, y, pen=pen(index, dashed=dashed), name=label)

        self.plot.addLine(y=1.0, pen=pen(4, width=1, dashed=True))
        self.plot.setYRange(0.0, max(1.4, float(np.max(y)) * 1.05))
