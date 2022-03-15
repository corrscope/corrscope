from typing import TYPE_CHECKING, List, Callable

from qtpy import QtWidgets as qw, QtCore as qc
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QShortcut

from corrscope.gui.util import find_ranges

qsp = qw.QSizePolicy

if TYPE_CHECKING:
    from corrscope.gui import ChannelModel


class ChannelTableView(qw.QTableView):
    def append_channels(self, wavs: List[str]):
        model: "ChannelModel" = self.model()

        begin_row = model.rowCount()
        count_rows = len(wavs)

        col = model.idx_of_key["wav_path"]

        # Insert N empty rows into model (mutates cfg.channels).
        model.insertRows(begin_row, count_rows)

        # Fill N empty rows with wav_path.
        for row, wav_path in enumerate(wavs, begin_row):
            index = model.index(row, col)
            model.setData(index, wav_path)

    def delete_selected(self):
        model: "ChannelModel" = self.model()
        rows = self.selected_rows()
        row_ranges = find_ranges(rows)

        for first_row, nrow in reversed(list(row_ranges)):
            model.removeRows(first_row, nrow)

    def on_channel_up(self):
        self.move_selection(-1)

    def on_channel_down(self):
        self.move_selection(1)

    def move_selection(self, delta: int):
        model: "ChannelModel" = self.model()
        rows = self.selected_rows()
        row_ranges = find_ranges(rows)

        # If we hit the end, cancel all other moves.
        # If moving up, move top first.
        if delta > 0:
            # If moving down, move bottom first.
            row_ranges = reversed(list(row_ranges))

        parent = qc.QModelIndex()
        for first_row, nrow in row_ranges:
            if delta > 0:
                dest_row = first_row + nrow + delta
            else:
                dest_row = first_row + delta

            if not model.moveRows(parent, first_row, nrow, parent, dest_row):
                break

    def selected_rows(self) -> List[int]:
        sel: qc.QItemSelectionModel = self.selectionModel()
        inds: List[qc.QModelIndex] = sel.selectedIndexes()
        rows: List[int] = sorted({ind.row() for ind in inds})
        return rows


class ShortcutButton(qw.QPushButton):
    scoped_shortcut: QShortcut

    def add_shortcut(self, scope: qw.QWidget, shortcut: str) -> None:
        """Adds shortcut and tooltip."""
        self.scoped_shortcut = new_shortcut(shortcut, scope, self.click)

        parsed_keys: QKeySequence = self.scoped_shortcut.key()
        self.setToolTip(parsed_keys.toString(QKeySequence.NativeText))


def new_shortcut(shortcut: str, scope: qw.QWidget, slot: Callable) -> qw.QShortcut:
    parsed_keys = QKeySequence(shortcut, QKeySequence.PortableText)

    scoped_shortcut = qw.QShortcut(parsed_keys, scope)
    scoped_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
    scoped_shortcut.activated.connect(slot)
    return scoped_shortcut


class TabWidget(qw.QTabWidget):
    def __init__(self, *args, **kwargs):
        qw.QTabWidget.__init__(self, *args, **kwargs)

        new_shortcut("ctrl+tab", self, self.next_tab)
        new_shortcut("ctrl+pgDown", self, self.next_tab)

        new_shortcut("ctrl+shift+tab", self, self.prev_tab)
        new_shortcut("ctrl+pgUp", self, self.prev_tab)

    def next_tab(self):
        self.setCurrentIndex((self.currentIndex() + 1) % self.count())

    def prev_tab(self):
        self.setCurrentIndex((self.currentIndex() - 1) % self.count())


class VerticalScrollArea(qw.QScrollArea):
    def __init__(self, parent):
        qw.QScrollArea.__init__(self, parent)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.horizontalScrollBar().setEnabled(False)

        # If removed, you will get unused space to the right and bottom.
        self.setWidgetResizable(True)

        # Only allow expanding, not shrinking.
        self.setSizePolicy(qsp(qsp.Minimum, qsp.Minimum))
