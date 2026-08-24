"""The rail, drawn as the pipeline it actually is (spec section 10.1).

The rail used to be a list of nine names, six of which said ``(not yet)``. That
is a true statement and a useless one. It tells the user that most of the
application is shut without telling them what shuts it, what would open it, or
that the nine names are a sequence at all rather than nine unrelated screens
somebody happened to disable.

So each step draws three things:

* **Its number, in order, joined to its neighbours by a line.** Nine numbered
  discs on a thread read as one process; nine labels read as a menu. The order
  is the tool's whole argument about how tuning works -- you look at the noise
  before you identify, you identify before you design -- and the rail is where
  that argument is made.
* **What the step is for**, in five or six words. A user who has never opened
  this program does not know what "Segment" means, and the place to say it is
  next to the word rather than in a manual.
* **Why it is not open yet**, when it is not. "Needs a log" and "Needs the
  analysis" are different problems with different fixes, and a user who is told
  which one they have can act on it. ``(not yet)`` sends them hunting.

Steps the user has already looked at get a tick, which makes the rail a record
of where they have been as well as a map of where they can go.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PySide6.QtCore import QModelIndex, QPersistentModelIndex, QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)

from rotorid.gui.theme import Palette, palette

__all__ = ["PipelineRail", "Step", "StepState"]

#: Where a step is in its life. ``ready`` and ``reviewed`` are both open; the
#: difference is only whether the user has been there, which is worth showing and
#: is not worth gating anything on.
StepState = Literal["waiting", "ready", "running", "reviewed"]

#: What Qt hands a delegate. Spelled out because the base class accepts either
#: and mypy holds an override to the wider of the two.
_Index = QModelIndex | QPersistentModelIndex

_ROW_HEIGHT = 58
_DISC = 24
_LEFT = 12
_TEXT_LEFT = _LEFT + _DISC + 12


@dataclass(frozen=True, slots=True)
class Step:
    """One rail entry: what it is, where it stands, and why."""

    name: str
    #: What this step does, in a handful of words.
    blurb: str
    state: StepState
    #: Why it is waiting, or what it is doing. Empty when the name says enough.
    note: str = ""

    @property
    def open(self) -> bool:
        return self.state != "waiting"


class _StepDelegate(QStyledItemDelegate):
    """Paints one step. Everything, including the row background.

    A delegate that defers the background to the stylesheet and then draws on top
    of it gets the selection wrong on exactly one platform, which is the sort of
    bug that is reported as "it looks broken on my machine" and never reproduced.
    """

    def __init__(self, theme: Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.theme = theme

    def sizeHint(self, option: QStyleOptionViewItem, index: _Index) -> QSize:
        return QSize(200, _ROW_HEIGHT)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: _Index) -> None:
        step = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(step, Step):  # pragma: no cover - defensive
            super().paint(painter, option, index)
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = option.rect
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        self._background(painter, rect, selected=selected, hovered=hovered)
        self._thread(painter, rect, index)
        self._disc(painter, rect, step, index.row(), selected=selected)
        self._text(painter, rect, step, selected=selected)
        painter.restore()

    # ------------------------------------------------------------------ #

    def _background(self, painter: QPainter, rect: QRect, *, selected: bool, hovered: bool) -> None:
        if not (selected or hovered):
            return
        # Selection is the louder of the two, because "where am I" is a question
        # the user asks constantly and "what would I get if I clicked" is one
        # they are already answering with the pointer.
        fill = self.theme.accent_soft if selected else self.theme.surface
        edge = self.theme.accent if selected else self.theme.grid
        painter.setPen(QPen(QColor(edge), 1))
        painter.setBrush(QColor(fill))
        painter.drawRoundedRect(rect.adjusted(3, 2, -4, -2), 8, 8)

    def _thread(self, painter: QPainter, rect: QRect, index: _Index) -> None:
        """The line joining one disc to the next. This is what makes it a sequence."""
        model = index.model()
        rows = model.rowCount() if model is not None else 0
        centre_x = rect.left() + _LEFT + _DISC // 2
        centre_y = rect.top() + rect.height() // 2
        painter.setPen(QPen(QColor(self.theme.grid), 2))
        if index.row() > 0:
            painter.drawLine(centre_x, rect.top(), centre_x, centre_y - _DISC // 2 - 2)
        if index.row() < rows - 1:
            painter.drawLine(centre_x, centre_y + _DISC // 2 + 2, centre_x, rect.bottom())

    def _disc(
        self, painter: QPainter, rect: QRect, step: Step, row: int, *, selected: bool
    ) -> None:
        disc = QRect(
            rect.left() + _LEFT,
            rect.top() + (rect.height() - _DISC) // 2,
            _DISC,
            _DISC,
        )
        if step.state == "waiting":
            fill, ink, edge = self.theme.background, self.theme.muted, self.theme.grid
        elif step.state == "running":
            band = self.theme.band("info")
            fill, ink, edge = band.background, band.foreground, band.rule
        elif selected:
            fill, ink, edge = self.theme.accent, self.theme.accent_text, self.theme.accent
        elif step.state == "reviewed":
            done = self.theme.band("good")
            fill, ink, edge = done.background, done.foreground, done.rule
        else:
            fill, ink, edge = self.theme.surface, self.theme.text, self.theme.grid

        painter.setBrush(QColor(fill))
        painter.setPen(QPen(QColor(edge), 1.5))
        painter.drawEllipse(disc)

        # A tick for done, the number otherwise. Never colour alone: the tick is
        # a shape, which survives being seen by a reader who cannot separate the
        # green from the grey.
        painter.setPen(QPen(QColor(ink), 2))
        if step.state == "reviewed":
            self._tick(painter, disc)
        else:
            font = QFont(painter.font())
            font.setPointSizeF(max(7.5, font.pointSizeF() - 1.5))
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(disc, Qt.AlignmentFlag.AlignCenter, str(row + 1))

    @staticmethod
    def _tick(painter: QPainter, disc: QRect) -> None:
        x, y, w, h = disc.x(), disc.y(), disc.width(), disc.height()
        painter.drawLine(
            round(x + 0.28 * w), round(y + 0.52 * h), round(x + 0.44 * w), round(y + 0.70 * h)
        )
        painter.drawLine(
            round(x + 0.44 * w), round(y + 0.70 * h), round(x + 0.74 * w), round(y + 0.32 * h)
        )

    def _text(self, painter: QPainter, rect: QRect, step: Step, *, selected: bool) -> None:
        left = rect.left() + _TEXT_LEFT
        width = rect.width() - _TEXT_LEFT - 10

        name_font = QFont(painter.font())
        name_font.setBold(selected or step.open)
        name_font.setPointSizeF(max(8.0, QFont().pointSizeF()))
        painter.setFont(name_font)
        painter.setPen(QColor(self.theme.text if step.open else self.theme.muted))
        painter.drawText(
            QRect(left, rect.top() + 10, width, 18),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            step.name,
        )

        note_font = QFont(painter.font())
        note_font.setBold(False)
        note_font.setPointSizeF(max(7.0, name_font.pointSizeF() - 1.5))
        painter.setFont(note_font)
        painter.setPen(QColor(self._note_colour(step)))
        painter.drawText(
            QRect(left, rect.top() + 28, width, 18),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            step.note or step.blurb,
        )

    def _note_colour(self, step: Step) -> str:
        if step.state == "running":
            return self.theme.band("info").rule
        return self.theme.muted


class PipelineRail(QListWidget):
    """The nine steps, in order, each saying where it stands.

    Still a :class:`QListWidget` underneath, so selection, keyboard navigation
    and accessibility come from the toolkit rather than from a hand-rolled
    imitation of it. Only the painting is ours.
    """

    def __init__(self, theme: Palette | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.theme = theme if theme is not None else palette("light")
        self.setObjectName("Rail")
        self.setMouseTracking(True)
        self.setUniformItemSizes(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setItemDelegate(_StepDelegate(self.theme, self))
        self.setFixedWidth(262)

    def set_theme(self, theme: Palette) -> None:
        """Repaint in the other palette. The delegate holds the colours, so it is replaced."""
        self.theme = theme
        self.setItemDelegate(_StepDelegate(theme, self))
        self.viewport().update()

    def set_steps(self, steps: tuple[Step, ...]) -> None:
        """Rebuild, or update in place when the shape has not changed.

        Updating in place matters: rebuilding the list drops the selection, and a
        rail that jumps back to step one every time the analysis reports progress
        is worse than one that never updates at all.
        """
        if self.count() != len(steps):
            self.clear()
            for step in steps:
                item = QListWidgetItem(step.name)
                self.addItem(item)

        for row, step in enumerate(steps):
            item = self.item(row)
            item.setText(step.name)
            item.setData(Qt.ItemDataRole.UserRole, step)
            item.setToolTip(f"{step.name} -- {step.blurb}\n{step.note}".strip())
            flags = item.flags()
            item.setFlags(
                flags | Qt.ItemFlag.ItemIsEnabled
                if step.open
                else flags & ~Qt.ItemFlag.ItemIsEnabled
            )
        self.viewport().update()

    def step(self, row: int) -> Step | None:
        item = self.item(row)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return value if isinstance(value, Step) else None
