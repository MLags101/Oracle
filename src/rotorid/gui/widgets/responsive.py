"""Cards that sit side by side when there is room, and stack when there is not.

Qt has no equivalent of a CSS grid that reflows, and a ``QHBoxLayout`` of two
cards has a minimum width equal to the sum of theirs. Put two information cards
in one and the page acquires a floor -- below it the window grows a horizontal
scrollbar, which is the worst way to read anything, and the two cards are read by
scrolling left and right between them.

So the row is told how much room it needs to keep the columns, and takes the
decision itself on every resize. Below the threshold it becomes one column and
the page simply gets taller, which it can afford, because every stage scrolls.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QGridLayout, QLayout, QLayoutItem, QSplitter, QWidget

__all__ = ["FlowLayout", "ResponsiveRow", "ResponsiveSplitter"]


class ResponsiveRow(QWidget):
    """A row of equal-weight widgets that collapses to a column when squeezed."""

    def __init__(
        self,
        *,
        threshold: int,
        spacing: int = 14,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        #: The width below which the columns are not worth keeping. Chosen per
        #: use from what the content needs, not globally: two dense tables want
        #: more room than two short paragraphs.
        self._threshold = threshold
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(spacing)
        self._grid.setVerticalSpacing(spacing)
        self._widgets: list[QWidget] = []
        self._columns = 0

    def add(self, widget: QWidget) -> None:
        self._widgets.append(widget)
        self._arrange(force=True)

    def minimumSizeHint(self) -> QSize:
        """As narrow as the widest single card, whatever the current arrangement.

        Without this the row answers for the layout it happens to be in, which is
        a deadlock dressed as a size: laid out in three columns it reports a wide
        minimum, the parent therefore refuses to get narrow, and the row never
        gets the resize that would have collapsed it to one column. Reporting
        what it *can* shrink to is both the truthful answer and the one that lets
        the collapse happen.
        """
        hint = super().minimumSizeHint()
        if not self._widgets:
            return hint
        width = max(widget.minimumSizeHint().width() for widget in self._widgets)
        return QSize(width + self._grid.contentsMargins().left() * 2, hint.height())

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._arrange()

    def _arrange(self, *, force: bool = False) -> None:
        if not self._widgets:
            return
        columns = len(self._widgets) if self.width() >= self._threshold else 1
        if columns == self._columns and not force:
            return
        self._columns = columns

        for widget in self._widgets:
            self._grid.removeWidget(widget)
        for column in range(len(self._widgets)):
            self._grid.setColumnStretch(column, 0)

        for index, widget in enumerate(self._widgets):
            row, column = (0, index) if columns > 1 else (index, 0)
            self._grid.addWidget(widget, row, column)
            self._grid.setColumnStretch(column, 1)


class ResponsiveSplitter(QSplitter):
    """Controls beside the plots when there is room, above them when there is not.

    The two working stages -- Filters and Design -- are a column of controls next
    to a column of plots, and both halves have a width below which they stop
    being readable rather than merely getting smaller: a gain table reading
    "Curren", an explain button reading "h". Enforcing those floors in a
    horizontal splitter gives the stage a combined floor of some six hundred
    pixels, which a 1200-pixel window with the findings dock open does not have.

    Turning the splitter on its side costs nothing, because the page scrolls.
    The user gets the controls at full width with the plots underneath, which is
    the layout every narrow window wants anyway.
    """

    def __init__(
        self,
        *,
        threshold: int,
        sizes: tuple[int, int],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._threshold = threshold
        self._sizes = sizes

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        wanted = (
            Qt.Orientation.Horizontal
            if self.width() >= self._threshold
            else Qt.Orientation.Vertical
        )
        if self.orientation() != wanted:
            self.setOrientation(wanted)
            if wanted is Qt.Orientation.Horizontal:
                self.setSizes(list(self._sizes))

    def minimumSizeHint(self) -> QSize:
        """As narrow as its widest child. See :meth:`ResponsiveRow.minimumSizeHint`.

        Same deadlock, same answer: while it is horizontal the honest sum of its
        children keeps the parent from ever handing it a width narrow enough to
        make it turn vertical.
        """
        hint = super().minimumSizeHint()
        widths = [
            child.minimumSizeHint().width()
            for child in (self.widget(i) for i in range(self.count()))
            if child is not None
        ]
        return QSize(max(widths, default=hint.width()), hint.height())


class FlowLayout(QLayout):
    """A row of controls that wraps onto the next line when it runs out of width.

    Qt has no such layout, and its absence is where narrow-window bugs come from:
    a ``QHBoxLayout`` of four buttons has a minimum width equal to the sum of all
    four, and nothing in the layout system will ever let it be narrower. One row
    of explain buttons was single-handedly holding the Filters stage above nine
    hundred pixels.

    This reports a minimum of the *widest single item* -- because that is what it
    can genuinely shrink to -- and answers ``heightForWidth`` with however many
    lines that width implies. Adapted from Qt's own flow layout example, which is
    the reference implementation everyone ends up rewriting.
    """

    def __init__(self, parent: QWidget | None = None, spacing: int = 8) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    # -- the five methods QLayout requires a subclass to provide ---------- #

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    # -- sizing ----------------------------------------------------------- #

    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._lay_out(QRect(0, 0, width, 0), place=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._lay_out(rect, place=True)

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _lay_out(self, rect: QRect, *, place: bool) -> int:
        """Place the items, or just measure how tall doing so would be."""
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, line_height = area.x(), area.y(), 0

        for item in self._items:
            hint = item.sizeHint()
            if x + hint.width() > area.right() + 1 and line_height > 0:
                x = area.x()
                y += line_height + self.spacing()
                line_height = 0
            if place:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self.spacing()
            line_height = max(line_height, hint.height())

        return y + line_height - rect.y() + margins.bottom()
