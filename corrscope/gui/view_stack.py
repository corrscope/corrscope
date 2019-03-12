from contextlib import contextmanager
from typing import *

import attr
from PyQt5.QtCore import QObject, Qt
from PyQt5.QtWidgets import *

from corrscope.util import obj_name

T = TypeVar("T")
ctx = Iterator
SomeQW = TypeVar("SomeQW", bound=QWidget)
WidgetOrLayout = TypeVar("WidgetOrLayout", bound=Union[QWidget, QLayout])


def new_widget_or_layout(
    item_type: Union[Type[WidgetOrLayout], str], parent: QWidget
) -> WidgetOrLayout:
    """Creates a widget or layout, for insertion into an existing layout.
    Do NOT use for filling a widget with a layout!"""
    if isinstance(item_type, str):
        return QLabel(item_type, parent)
    elif issubclass(item_type, QWidget):
        return item_type(parent)
    else:
        assert issubclass(item_type, QLayout)
        # new_widget_or_layout is used to add sublayouts, which do NOT have a parent.
        # Only widgets' root layouts have parents.
        return item_type(None)


@attr.dataclass
class StackFrame:
    widget: Optional[QWidget]
    layout: Optional[QLayout] = None
    parent: Optional["StackFrame"] = None

    def with_layout(self, layout: Optional[QLayout]):
        return attr.evolve(self, layout=layout)


class LayoutStack:
    def __init__(self, root: Optional[QWidget]):
        self._items = [StackFrame(root)]
        self._widget_to_label: Dict[QWidget, QLabel] = {}

    @contextmanager
    def push(self, item: T) -> ctx[T]:
        if isinstance(item, StackFrame):
            frame = item  # this branch never happens
        elif isinstance(item, QWidget):
            frame = StackFrame(item)
        elif isinstance(item, QLayout):
            frame = self.peek().with_layout(item)
        else:
            raise TypeError(obj_name(item))

        frame.parent = self.peek()
        self._items.append(frame)

        try:
            yield item
        finally:
            self._items.pop()

    def peek(self) -> StackFrame:
        return self._items[-1]

    @property
    def widget(self):
        return self.peek().widget

    @property
    def layout(self):
        return self.peek().layout

    @property
    def parent(self):
        return self.peek().parent


def set_layout(stack: LayoutStack, layout_type: Type[QLayout]) -> QLayout:
    layout = layout_type(stack.peek().widget)
    stack.peek().layout = layout
    return layout


def assert_peek(stack: LayoutStack, cls):
    assert isinstance(stack.widget, cls)


def central_widget(stack: LayoutStack, widget_type: Type[SomeQW] = QWidget):
    assert_peek(stack, QMainWindow)
    # do NOT orphan=True
    return _new_widget(stack, widget_type, exit_action="setCentralWidget")


def orphan_widget(stack: LayoutStack, widget_type: Type[SomeQW] = QWidget):
    return _new_widget(stack, widget_type, orphan=True)


@contextmanager
def append_widget(
    stack: LayoutStack, item_type: Type[WidgetOrLayout], layout_args: list = []
) -> ctx[WidgetOrLayout]:
    with _new_widget(stack, item_type) as item:
        yield item
    _insert_widget_or_layout(stack.layout, item, *layout_args)


# noinspection PyArgumentList
def _insert_widget_or_layout(layout: QLayout, item: WidgetOrLayout, *args, **kwargs):
    if isinstance(item, QWidget):
        layout.addWidget(item, *args, **kwargs)
    elif isinstance(item, QLayout):
        # QLayout and some subclasses (like QFormLayout) omit this method,
        # and will crash at runtime.
        layout.addLayout(item, *args, **kwargs)
    else:
        raise TypeError(item)


# Main window toolbars/menus


def set_menu_bar(stack: LayoutStack):
    assert_peek(stack, QMainWindow)
    return _new_widget(stack, QMenuBar, exit_action="setMenuBar")


# won't bother adding type hints that pycharm is too dumb to understand
def append_menu(stack: LayoutStack):
    assert_peek(stack, QMenuBar)
    return _new_widget(stack, QMenu, exit_action="addMenu")


def add_toolbar(stack: LayoutStack, area=Qt.TopToolBarArea):
    assert_peek(stack, QMainWindow)

    def _add_toolbar(parent: QMainWindow, toolbar):
        parent.addToolBar(area, toolbar)

    return _new_widget(stack, QToolBar, exit_action=_add_toolbar)


# Implementation


@contextmanager
def _new_widget(
    stack: LayoutStack,
    item_type: Type[WidgetOrLayout],
    orphan=False,
    exit_action: Union[Callable[[Any, Any], Any], str] = "",
) -> ctx[WidgetOrLayout]:
    """
    - Constructs item_type using parent.
    - Yields item_type.
    """

    if not orphan:
        parent = stack.widget
    else:
        parent = None

    with stack.push(new_widget_or_layout(item_type, parent)) as item:
        yield item

    real_parent = stack.widget
    if callable(exit_action):
        exit_action(real_parent, item)
    elif exit_action:
        getattr(real_parent, exit_action)(item)


def append_stretch(stack: LayoutStack):
    cast(QBoxLayout, stack.layout).addStretch()


Left = TypeVar("Left", bound=QWidget)
Right = TypeVar("Right", bound=Union[QWidget, QLayout])  # same as WidgetOrLayout


class _Both:
    def __bool__(self):
        return False


Both = _Both()


def widget_pair_inserter(append_widgets: Callable):
    @contextmanager
    def add_row_col(stack: LayoutStack, left_type, right_type, *, name=None):
        parent = stack.widget
        left = new_widget_or_layout(left_type, parent)
        left_is_label = isinstance(left, QLabel)

        if right_type is Both:
            right = Both
            push = left
        else:
            right = new_widget_or_layout(right_type, parent)
            push = right

        if name:
            (right or left).setObjectName(name)

        with stack.push(push):
            if right is Both:
                yield left
            elif left_is_label:
                yield right
            else:
                yield left, right

        append_widgets(stack.layout, left, right)
        if left_is_label:
            assert isinstance(left, QLabel)
            stack._widget_to_label[right] = left

    return add_row_col


def _add_row(layout: QFormLayout, left, right):
    assert isinstance(layout, QFormLayout), layout
    if right is Both:
        raise TypeError("Cannot add_row(QFormLayout, span=Both)")
    return layout.addRow(left, right)


add_row = widget_pair_inserter(_add_row)


def _add_grid_col(layout: QGridLayout, up, down):
    assert isinstance(layout, QGridLayout), layout
    col = layout.columnCount()

    """
    void QGridLayout::addWidget(
        QWidget *widget,
        int fromRow, int fromColumn, [int rowSpan, int columnSpan],
        Qt::Alignment alignment = Qt::Alignment()
    )
    """
    if down is Both:
        shape = lambda: [0, col, -1, 1]
        _insert_widget_or_layout(layout, up, *shape())
    else:
        shape = lambda row: [row, col]
        _insert_widget_or_layout(layout, up, *shape(0))
        _insert_widget_or_layout(layout, down, *shape(1))


add_grid_col = widget_pair_inserter(_add_grid_col)


@contextmanager
def add_tab(stack, widget_type: Type[SomeQW] = QWidget, label: str = "") -> ctx[SomeQW]:
    """
    - Constructs widget using parent.
    - Yields widget.
    """
    tabs: QTabWidget = stack.widget
    assert isinstance(tabs, QTabWidget), tabs

    with orphan_widget(stack, widget_type) as w:
        yield w
    tabs.addTab(w, label)


# After building a tree...
def set_attr_objectName(ui, stack: LayoutStack):
    """
    - Set objectName of all objects referenced by ui.
    - For all object $name added by add_row() or add_grid_col(),
        if $label was generated but not yielded
        setattr(ui.$name + "L" = $label)
    """
    widget_to_label = stack._widget_to_label

    for name, obj in dict(ui.__dict__).items():
        if not isinstance(obj, QObject):
            continue
        obj.setObjectName(name)
        if obj in widget_to_label:
            label = widget_to_label[obj]
            label_name = name + "L"
            label.setObjectName(label_name)
            ui.__dict__[label_name] = label
