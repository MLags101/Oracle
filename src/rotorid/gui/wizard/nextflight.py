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
    QVBoxLayout,
    QWidget,
)

from rotorid.core.types import FlightTestStage
from rotorid.gui.state import AppState
from rotorid.gui.theme import Palette
from rotorid.gui.widgets.layouts import clear
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
                "Next Flight",
                subtitle=(
                    "What to fly to confirm the change, in the order that keeps each "
                    "flight answerable on its own."
                ),
            )
        )

        # The shell already wraps every stage in a scroll area; a second one here
        # would fight it for the wheel.
        self._body = QWidget()
        self._items = QVBoxLayout(self._body)
        self._items.setContentsMargins(0, 0, 0, 0)
        self._items.setSpacing(12)
        layout.addWidget(self._body, 1)

        state.analysis_finished.connect(lambda *_: self.refresh())

    def refresh(self) -> None:
        clear(self._items)

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
    # Card headings are prose. Unwrapped, a title like "Flight 2: Rate loop I and
    # feed-forward" sets a minimum width the page can never give back.
    heading.setWordWrap(True)
    layout.addWidget(heading)
    layout.addWidget(_wrapped(text))
    return frame


def _wrapped(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(label.textInteractionFlags().TextSelectableByMouse)
    return label
