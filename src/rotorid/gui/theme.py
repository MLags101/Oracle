"""Colours and text conventions (spec section 10.6).

Three rules here are correctness rules rather than taste:

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
* **Every colour on screen comes from the palette in force**, not from a hex
  typed into the widget that needed it. A severity banner with a hard-coded dark
  background is unreadable the moment somebody opens the light theme, and that
  is the sort of thing nobody notices until a user photographs it.

The palette carries a severity *band* -- a background, a foreground and a rule
colour -- as well as the single accent colour, because the loud things in this
application are blocks of text rather than dots on a chart, and a block of text
needs all three.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from PySide6.QtWidgets import QLabel

from rotorid.core.types import Severity

__all__ = [
    "SERIES",
    "SEVERITY_MARK",
    "Band",
    "Mode",
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
    """The line colour for a severity. For plots and for one-line labels."""
    return _SEVERITY_COLOUR[severity]


@dataclass(frozen=True, slots=True)
class Band:
    """A severity rendered as a block of text rather than as a dot on a chart."""

    background: str
    foreground: str
    rule: str

    def style(self, *, radius: int = 6) -> str:
        """A Qt stylesheet fragment for a banner in this severity, without padding.

        Padding is deliberately absent -- see :meth:`apply`.
        """
        return (
            f"background: {self.background}; color: {self.foreground};"
            f" border: 1px solid {self.rule}; border-left: 4px solid {self.rule};"
            f" border-radius: {radius}px;"
        )

    def apply(
        self, label: QLabel, *, radius: int = 6, padding: int = 10, bold: bool = False
    ) -> None:
        """Paint ``label`` as a banner in this severity, breathing room included.

        The room comes from the label's contents margins rather than from CSS
        padding, and that is a correctness choice rather than a stylistic one.
        Qt sizes wrapped text by asking the label how tall it is at a given
        width, and that answer is computed from the layout's own geometry --
        stylesheet padding is applied later, when the label paints. A banner with
        CSS padding therefore reports a height some twenty pixels short of what
        it draws, and the last line of every long finding is sliced off.
        Contents margins are part of the measurement, so they cost exactly what
        they claim to.
        """
        weight = " font-weight: 600;" if bold else ""
        label.setStyleSheet(self.style(radius=radius) + weight)
        label.setContentsMargins(padding + 4, padding, padding, padding)


Mode = Literal["light", "dark"]

#: Tinted blocks with dark ink. Light-theme banners have to stay legible beside
#: white cards, so the tint is shallow and the ink carries the contrast.
_LIGHT_BANDS: Final[dict[Severity, Band]] = {
    "blocker": Band("#fdece4", "#8a3a10", "#d55e00"),
    "warning": Band("#fdf4e3", "#785110", "#e69f00"),
    "info": Band("#e8f3fb", "#134f6d", "#0072b2"),
    "good": Band("#e4f4ee", "#0e5940", "#009e73"),
}

#: Deep blocks with light ink, at the same hues. Holding the hue constant across
#: the two themes is what lets a user who switches recognise the same warning.
_DARK_BANDS: Final[dict[Severity, Band]] = {
    "blocker": Band("#3a1c10", "#ffbb98", "#d55e00"),
    "warning": Band("#382e13", "#f4d089", "#e69f00"),
    "info": Band("#12303f", "#a8dbf7", "#56b4e9"),
    "good": Band("#0e3129", "#82dbc0", "#009e73"),
}


@dataclass(frozen=True, slots=True)
class Palette:
    """Everything the widgets need to paint themselves in one of the two modes."""

    mode: Mode
    #: The window itself, behind everything.
    background: str
    #: Cards and plots, one step forward from the background.
    surface: str
    #: Rails, headers and table stripes -- one step *back*, so a card sitting on
    #: top of one still reads as raised.
    surface_alt: str
    text: str
    muted: str
    #: Hairlines: borders, plot grids, table rules. One colour for all three, so
    #: nothing on screen is separated by a line that does not match its
    #: neighbours.
    grid: str
    accent: str
    #: The accent as a block fill, with ink that stays legible on it, and the
    #: same hue at a whisper for hovers and selections.
    accent_text: str
    accent_soft: str
    bands: Mapping[Severity, Band]

    def band(self, severity: Severity) -> Band:
        return self.bands[severity]

    def stylesheet(self) -> str:
        """Application-wide Qt stylesheet.

        Larger than it used to be, and deliberately so. The previous version
        styled four widgets and left the rest to the platform, which sounds
        respectful and reads as unfinished: a default list view beside a
        hand-styled card looks like a screen somebody stopped working on. What is
        set here is structure -- surfaces, hairlines, spacing, one accent -- and
        not a reimplementation of the toolkit. Controls still behave and animate
        the way the platform's own do.
        """
        return f"""
        QWidget {{
            background: {self.background};
            color: {self.text};
            font-size: 13px;
        }}
        QMainWindow::separator {{ background: {self.grid}; width: 1px; height: 1px; }}
        /* Text sits on whatever it was put on. Without this the ``QWidget`` rule
           above paints the window colour behind every label, which turns each
           line of type inside a white card into a grey brick. */
        QLabel, QCheckBox, QRadioButton {{ background: transparent; }}

        /* ---- text roles -------------------------------------------------- */
        QLabel#Title {{ font-size: 21px; font-weight: 600; }}
        QLabel#Heading {{ font-size: 16px; font-weight: 600; }}
        QLabel#Subheading {{ font-size: 13px; font-weight: 600; }}
        QLabel#Muted {{ color: {self.muted}; }}
        QLabel#Eyebrow {{
            color: {self.muted}; font-size: 11px; font-weight: 600;
        }}

        /* ---- surfaces ---------------------------------------------------- */
        QFrame#Card, QGroupBox {{
            background: {self.surface};
            border: 1px solid {self.grid};
            border-radius: 8px;
        }}
        QGroupBox {{ margin-top: 14px; padding-top: 10px; }}
        QGroupBox::title {{
            subcontrol-origin: margin; left: 10px; padding: 0 4px;
            color: {self.muted}; font-weight: 600;
        }}
        QFrame#Rule {{ background: {self.grid}; max-height: 1px; border: none; }}
        QFrame#DropZone {{
            background: {self.surface};
            border: 2px dashed {self.grid};
            border-radius: 12px;
        }}

        /* ---- buttons ----------------------------------------------------- */
        QPushButton {{
            background: {self.surface}; border: 1px solid {self.grid};
            border-radius: 6px; padding: 6px 14px;
        }}
        QPushButton:hover {{ border-color: {self.accent}; }}
        QPushButton:pressed {{ background: {self.accent_soft}; }}
        QPushButton:disabled {{ color: {self.muted}; border-color: {self.grid}; }}
        QPushButton:checked {{
            background: {self.accent_soft}; border-color: {self.accent};
            font-weight: 600;
        }}
        QPushButton#Primary {{
            background: {self.accent}; color: {self.accent_text};
            border: 1px solid {self.accent}; font-weight: 600; padding: 7px 18px;
        }}
        QPushButton#Primary:hover {{ border-color: {self.text}; }}
        QPushButton#Primary:disabled {{
            background: {self.surface_alt}; color: {self.muted};
            border-color: {self.grid};
        }}
        QPushButton#Link {{
            background: transparent; border: none; color: {self.accent};
            padding: 2px 4px; text-decoration: underline;
        }}

        /* ---- the toolbar, which is where the verbs live ------------------- */
        QToolBar {{
            background: {self.surface_alt};
            border: none; border-bottom: 1px solid {self.grid};
            padding: 6px 10px; spacing: 8px;
        }}
        QToolBar QToolButton {{
            background: transparent; border: 1px solid transparent;
            border-radius: 6px; padding: 5px 10px;
        }}
        QToolBar QToolButton:hover {{
            background: {self.surface}; border-color: {self.grid};
        }}
        QToolBar QToolButton:disabled {{ color: {self.muted}; }}
        QToolBar QToolButton:checked {{
            background: {self.accent_soft}; border-color: {self.accent};
        }}
        QToolBar::separator {{ background: {self.grid}; width: 1px; margin: 4px 6px; }}

        QMenuBar {{ background: {self.surface_alt}; border-bottom: 1px solid {self.grid}; }}
        QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
        QMenuBar::item:selected {{ background: {self.accent_soft}; border-radius: 4px; }}
        QMenu {{ background: {self.surface}; border: 1px solid {self.grid}; padding: 4px; }}
        QMenu::item {{ padding: 6px 24px 6px 12px; border-radius: 4px; }}
        QMenu::item:selected {{ background: {self.accent_soft}; }}
        QMenu::separator {{ height: 1px; background: {self.grid}; margin: 4px 8px; }}

        /* ---- the rail ---------------------------------------------------- */
        QListWidget#Rail {{
            background: {self.surface_alt};
            border: none; border-right: 1px solid {self.grid};
            outline: none; padding: 6px;
        }}
        QListWidget#Rail::item {{ border-radius: 8px; margin: 1px 0; }}
        QListWidget#Rail::item:selected {{ background: {self.surface}; }}
        QListWidget#Rail::item:hover:!selected {{ background: {self.accent_soft}; }}

        /* ---- docks ------------------------------------------------------- */
        QDockWidget::title {{
            background: {self.surface_alt}; padding: 7px 10px;
            border-bottom: 1px solid {self.grid};
            font-weight: 600;
        }}

        /* ---- tables and trees --------------------------------------------- */
        QHeaderView::section {{
            background: {self.surface_alt}; color: {self.muted};
            border: none; border-bottom: 1px solid {self.grid};
            padding: 6px 8px; font-weight: 600;
        }}
        QTableView, QTreeView, QTableWidget, QTreeWidget {{
            background: {self.surface};
            alternate-background-color: {self.surface_alt};
            border: 1px solid {self.grid}; border-radius: 8px;
            gridline-color: {self.grid};
            selection-background-color: {self.accent_soft};
            selection-color: {self.text};
        }}
        QTableView::item, QTreeView::item {{ padding: 4px 6px; }}

        /* ---- progress, scrollbars, inputs --------------------------------- */
        QProgressBar {{
            background: {self.surface_alt}; border: 1px solid {self.grid};
            border-radius: 6px; height: 10px; text-align: center; color: {self.muted};
        }}
        QProgressBar::chunk {{ background: {self.accent}; border-radius: 5px; }}

        QScrollArea {{ border: none; background: {self.background}; }}
        QScrollBar:vertical {{ background: transparent; width: 12px; margin: 0; }}
        QScrollBar::handle:vertical {{
            background: {self.grid}; border-radius: 6px; min-height: 28px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {self.muted}; }}
        QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 0; }}
        QScrollBar::handle:horizontal {{
            background: {self.grid}; border-radius: 6px; min-width: 28px;
        }}
        QScrollBar::handle:horizontal:hover {{ background: {self.muted}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
            background: {self.surface}; border: 1px solid {self.grid};
            border-radius: 6px; padding: 5px 8px;
            selection-background-color: {self.accent_soft};
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {self.accent};
        }}
        QSlider::groove:horizontal {{
            background: {self.surface_alt}; height: 5px; border-radius: 3px;
        }}
        QSlider::sub-page:horizontal {{ background: {self.accent}; border-radius: 3px; }}
        QSlider::handle:horizontal {{
            background: {self.surface}; border: 2px solid {self.accent};
            width: 14px; margin: -6px 0; border-radius: 9px;
        }}
        /* Both indicators are drawn here rather than left to the platform.
           Styling any property of a check box or radio button switches Qt to the
           stylesheet box model for the whole widget, and an indicator with no
           rule of its own then draws as nothing at all -- which is how three
           radio buttons end up looking like three lines of text. */
        QCheckBox, QRadioButton {{ spacing: 8px; padding: 2px 0; }}
        QCheckBox::indicator, QRadioButton::indicator {{
            width: 15px; height: 15px;
            background: {self.surface}; border: 1px solid {self.muted};
        }}
        QCheckBox::indicator {{ border-radius: 4px; }}
        QRadioButton::indicator {{ border-radius: 8px; }}
        QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
            border-color: {self.accent};
        }}
        /* Filled means on. No tick glyph: Qt's ``image:`` resolves file paths
           rather than data URIs, and shipping a bitmap for two checkboxes is not
           worth an asset directory. The state is carried by fill *and* by the
           label sitting beside it, which is enough for a control that is never
           the only thing saying what it says. */
        QCheckBox::indicator:checked {{
            background: {self.accent}; border: 4px solid {self.surface};
            border-radius: 4px; outline: 1px solid {self.accent};
        }}
        QRadioButton::indicator:checked {{
            background: {self.accent}; border: 4px solid {self.surface};
            border-radius: 8px; outline: 1px solid {self.accent};
        }}
        QCheckBox:disabled, QRadioButton:disabled {{ color: {self.muted}; }}
        QToolTip {{
            background: {self.surface}; color: {self.text};
            border: 1px solid {self.grid}; padding: 6px;
        }}
        QSplitter::handle {{ background: {self.grid}; }}
        QStatusBar {{ background: {self.surface_alt}; border-top: 1px solid {self.grid}; }}
        QStatusBar::item {{ border: none; }}
        """


_LIGHT = Palette(
    mode="light",
    background="#f3f4f7",
    surface="#ffffff",
    surface_alt="#e9ebf0",
    text="#14161a",
    muted="#5f6672",
    grid="#d5d9e0",
    accent="#0072b2",
    accent_text="#ffffff",
    accent_soft="#dcebf6",
    bands=_LIGHT_BANDS,
)

_DARK = Palette(
    mode="dark",
    background="#101319",
    surface="#191d25",
    surface_alt="#1f242e",
    text="#e8eaef",
    muted="#98a1b0",
    grid="#2c323d",
    accent="#56b4e9",
    accent_text="#0b1016",
    accent_soft="#1d3444",
    bands=_DARK_BANDS,
)


def palette(mode: Mode = "light") -> Palette:
    return _DARK if mode == "dark" else _LIGHT
