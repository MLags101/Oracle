"""Stage 8: Next flight (spec sections 8.2 and 10.2).

The plan as something to take to the field: what to change, what to watch for in
the air, what to look for in the log afterwards, and how to set up the in-flight
tuning knob so a gain can be backed off without landing.

The last part is not decoration. The most useful safety property a tuning flight
can have is the ability to reduce a gain in the air the moment it starts to ring,
and a pilot who has not set that up before taking off cannot do it.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from rotorid.core.types import FlightTestStage
from rotorid.gui.state import AppState
from rotorid.gui.wizard.base import StageWidget

__all__ = ["NextFlightStage"]

_KNOB = (
    "Set the in-flight tuning knob before you take off:\n"
    "    RC*_OPTION or TUNE = 4 puts the rate roll/pitch P gain on a transmitter "
    "knob, TUNE = 21 puts rate roll/pitch D on it, and TUNE = 58 adjusts the "
    "SYSTEMID magnitude.\n"
    "    Set TUNE_MIN and TUNE_MAX to bracket the recommended value -- typically "
    "half of it to a little above it -- so the knob can always take you back to "
    "something you have already flown.\n\n"
    "If the aircraft starts to oscillate, wind the knob down before doing anything "
    "else. Landing an oscillating multirotor is harder than reducing the gain that "
    "is causing it."
)


class NextFlightStage(StageWidget):
    """The plan, in the order it should be flown, with the setup to fly it safely."""

    title = "Next Flight"

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)

        layout = QVBoxLayout(self)
        heading = QLabel("Next flight")
        heading.setObjectName("Heading")
        layout.addWidget(heading)

        self._body = QWidget()
        self._items = QVBoxLayout(self._body)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._body)
        layout.addWidget(scroll, 1)

        state.analysis_finished.connect(lambda *_: self.refresh())

    def refresh(self) -> None:
        while self._items.count():
            item = self._items.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()

        result = self.state.result
        plan = result.session.next_steps if result is not None else None
        if plan is None or not plan.stages:
            self._items.addWidget(_wrapped("Nothing to fly yet -- run the analysis first."))
            return

        self._items.addWidget(_wrapped(plan.preamble))
        self._items.addWidget(_card(_KNOB, "Before you take off"))
        for stage in plan.stages:
            self._items.addWidget(_flight_card(stage))
        self._items.addStretch(1)


def _flight_card(stage: FlightTestStage) -> QWidget:
    lines = [
        "Set:",
        *(f"    {name} = {value:g}" for name, value in sorted(stage.changes.items())),
        "",
        "Watch for:",
        *(f"    - {item}" for item in stage.watch_in_flight),
        "",
        "Then, in the log:",
        *(f"    - {item}" for item in stage.check_in_log),
    ]
    return _card("\n".join(lines), f"Flight {stage.index}: {stage.title}")


def _card(text: str, title: str) -> QWidget:
    frame = QFrame()
    frame.setObjectName("Card")
    layout = QVBoxLayout(frame)
    heading = QLabel(title)
    heading.setObjectName("Heading")
    layout.addWidget(heading)
    layout.addWidget(_wrapped(text))
    return frame


def _wrapped(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(label.textInteractionFlags().TextSelectableByMouse)
    return label
