"""Colours and text conventions (spec section 10.6).

Two rules here are correctness rules rather than taste:

* **Nothing is distinguished by red versus green alone.** Roughly one man in
  twelve cannot separate them, and this tool uses colour to say whether a
  parameter is present and whether a finding blocks -- both of which have to
  survive being seen by that reader. Severity therefore carries a shape or a
  word as well as a colour, and the series palette is Okabe-Ito, which is
  distinguishable under all common forms of colour blindness.
* **Frequencies are shown in Hz.** The whole design is done in rad/s internally
  and every number that reaches the screen is converted, because the parameters
  the user will type into a ground station are in Hz and a unit switch between
  the two is where a factor of 2*pi goes missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from rotorid.core.types import Severity

__all__ = [
    "SERIES",
    "SEVERITY_MARK",
    "Palette",
    "palette",
    "severity_colour",
]

#: Okabe-Ito, the standard colourblind-safe qualitative palette. Order matters:
#: the first three are the ones used for measured / modelled / predicted, which
#: are the three traces most often on one axis together.
SERIES: Final[tuple[str, ...]] = (
    "#0072b2",  # blue    -- measured
    "#e69f00",  # orange  -- modelled
    "#009e73",  # green   -- predicted / recommended
    "#cc79a7",  # pink    -- baseline
    "#56b4e9",  # sky     -- secondary
    "#d55e00",  # vermillion
    "#f0e442",  # yellow
)

#: A word, not just a colour. See the module docstring.
SEVERITY_MARK: Final[dict[Severity, str]] = {
    "blocker": "BLOCKS",
    "warning": "WARN",
    "info": "NOTE",
    "good": "OK",
}

_SEVERITY_COLOUR: Final[dict[Severity, str]] = {
    "blocker": "#d55e00",
    "warning": "#e69f00",
    "info": "#56b4e9",
    "good": "#009e73",
}


def severity_colour(severity: Severity) -> str:
    return _SEVERITY_COLOUR[severity]


Mode = Literal["light", "dark"]


@dataclass(frozen=True, slots=True)
class Palette:
    """Everything the widgets need to paint themselves in one of the two modes."""

    mode: Mode
    background: str
    surface: str
    text: str
    muted: str
    grid: str
    accent: str

    def stylesheet(self) -> str:
        """Application-wide Qt stylesheet.

        Deliberately small: the aim is a consistent surface and text colour, not
        a restyled toolkit. Anything more elaborate stops matching the platform
        the user is actually on.
        """
        return f"""
        QWidget {{ background: {self.background}; color: {self.text}; }}
        QFrame#Card, QGroupBox {{ background: {self.surface}; border-radius: 6px; }}
        QLabel#Muted {{ color: {self.muted}; }}
        QLabel#Heading {{ font-size: 15px; font-weight: 600; }}
        QLabel#Subheading {{ font-size: 13px; font-weight: 600; }}
        QPushButton {{
            background: {self.surface}; border: 1px solid {self.grid};
            border-radius: 4px; padding: 5px 12px;
        }}
        QPushButton:hover {{ border-color: {self.accent}; }}
        QPushButton:disabled {{ color: {self.muted}; }}
        QHeaderView::section {{
            background: {self.surface}; color: {self.muted};
            border: none; border-bottom: 1px solid {self.grid}; padding: 4px;
        }}
        QTableView {{ gridline-color: {self.grid}; }}
        """


_LIGHT = Palette(
    mode="light",
    background="#fbfbfa",
    surface="#ffffff",
    text="#1a1a19",
    muted="#6b6b68",
    grid="#dcdcd8",
    accent="#0072b2",
)

_DARK = Palette(
    mode="dark",
    background="#16161a",
    surface="#1f1f24",
    text="#ececef",
    muted="#9a9aa2",
    grid="#33333a",
    accent="#56b4e9",
)


def palette(mode: Mode = "light") -> Palette:
    return _DARK if mode == "dark" else _LIGHT
