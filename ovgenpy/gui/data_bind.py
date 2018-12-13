import functools
from typing import Optional, List, Callable, Dict, Any

import attr
from PyQt5 import QtWidgets as qw, QtCore as qc
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget

from ovgenpy.util import obj_name, perr

__all__ = ['PresentationModel', 'map_gui', 'behead', 'rgetattr', 'rsetattr']


WidgetUpdater = Callable[[], None]
Attrs = Any


class PresentationModel:
    """ Key-value MVP presentation-model.

    Qt's built-in model-view framework expects all models to
    take the form of a database-style numbered [row, column] structure,
    whereas my model takes the form of a key-value struct exposed as a form.
    """

    # These fields are specific to each subclass, and assigned there.
    # Although less explicit, these can be assigned using __init_subclass__.
    combo_symbols: Dict[str, List[str]]
    combo_text: Dict[str, List[str]]

    def __init__(self, cfg: Attrs):
        self.cfg = cfg
        self.update_widget: Dict[str, WidgetUpdater] = {}

    def __getitem__(self, item):
        try:
            # Custom properties
            return getattr(self, item)
        except AttributeError:
            return rgetattr(self.cfg, item)

    def __setitem__(self, key, value):
        # Custom properties
        if hasattr(self, key):
            setattr(self, key, value)
        elif rhasattr(self.cfg, key):
            rsetattr(self.cfg, key, value)
        else:
            raise AttributeError(f'cannot set attribute {key} on {obj_name(self)}()')

    def set_cfg(self, cfg: Attrs):
        self.cfg = cfg
        for updater in self.update_widget.values():
            updater()


BIND_PREFIX = 'cfg__'

# TODO add tests for recursive operations
def map_gui(view: QWidget, model: PresentationModel):
    """
    Binding:
    - .ui <widget name="cfg__layout__nrows">
    - view.cfg__layout__nrows
    - model['layout__nrows']

    Only <widget>s starting with 'cfg__' will be bound.
    """

    widgets: List[QWidget] = view.findChildren(QWidget)  # dear pyqt, add generic mypy return types
    for widget in widgets:
        widget_name = widget.objectName()
        path = try_behead(widget_name, BIND_PREFIX)
        if path is not None:
            bind_widget(widget, model, path)


@functools.singledispatch
def bind_widget(widget: QWidget, model: PresentationModel, path: str):
    perr(widget, path)
    return


@bind_widget.register(qw.QLineEdit)
def _(widget, model: PresentationModel, path: str):
    direct_bind(widget, model, path, DirectBinding(
        set_widget=widget.setText,
        widget_changed=widget.textChanged,
        value_type=str,
    ))


@bind_widget.register(qw.QSpinBox)
def _(widget, model: PresentationModel, path: str):
    direct_bind(widget, model, path, DirectBinding(
        set_widget=widget.setValue,
        widget_changed=widget.valueChanged,
        value_type=int,
    ))


@bind_widget.register(qw.QDoubleSpinBox)
def _(widget, model: PresentationModel, path: str):
    direct_bind(widget, model, path, DirectBinding(
        set_widget=widget.setValue,
        widget_changed=widget.valueChanged,
        value_type=float,
    ))


@bind_widget.register(qw.QComboBox)
def _(widget, model: PresentationModel, path: str):
    combo_symbols = model.combo_symbols[path]
    combo_text = model.combo_text[path]
    symbol2idx = {}
    for i, symbol in enumerate(combo_symbols):
        symbol2idx[symbol] = i
        widget.addItem(combo_text[i])

    # combobox.index = model.attr
    def set_widget(symbol: str):
        combo_index = symbol2idx[symbol]
        widget.setCurrentIndex(combo_index)

    # model.attr = combobox.index
    def set_model(combo_index: int):
        assert isinstance(combo_index, int)
        model[path] = combo_symbols[combo_index]
    widget.currentIndexChanged.connect(set_model)

    direct_bind(widget, model, path, DirectBinding(
        set_widget=set_widget,
        widget_changed=None,
        value_type=None,
    ))


Signal = Any

@attr.dataclass
class DirectBinding:
    set_widget: Callable
    widget_changed: Optional[Signal]
    value_type: Optional[type]


def direct_bind(widget: QWidget, model: PresentationModel, path: str, bind: DirectBinding):
    try:
        def update_widget():
            """ Update the widget without triggering signals.

            When the presentation model updates dependent widget 1,
            the model (not widget 1) is responsible for updating other
            dependent widgets.
            """
            # TODO add option to send signals
            with qc.QSignalBlocker(widget):
                bind.set_widget(model[path])

        update_widget()

        # Allow widget to be updated by other events.
        model.update_widget[path] = update_widget

        # Allow model to be changed by widget.
        if bind.widget_changed is not None:
            @pyqtSlot(bind.value_type)
            def set_model(value):
                assert isinstance(value, bind.value_type)
                model[path] = value

            bind.widget_changed.connect(set_model)

        # QSpinBox.valueChanged may or may not be called with (str).
        # http://pyqt.sourceforge.net/Docs/PyQt5/signals_slots.html#connecting-slots-by-name
        # mentions connectSlotsByName(),
        # but we're using QSpinBox().valueChanged.connect() and my assert never fails.
        # Either way, @pyqtSlot(value_type) will ward off incorrect calls.

    except Exception:
        perr(widget)
        perr(path)
        raise


def try_behead(string: str, header: str) -> Optional[str]:
    if not string.startswith(header):
        return None
    return string[len(header):]


def behead(string: str, header: str) -> str:
    if not string.startswith(header):
        raise ValueError(f'{string} does not start with {header}')
    return string[len(header):]


DUNDER = '__'

# https://gist.github.com/wonderbeyond/d293e7a2af1de4873f2d757edd580288
def rgetattr(obj, dunder_delim_path: str, *default):
    """
    :param obj: Object
    :param dunder_delim_path: 'attr1__attr2__etc'
    :param default: Optional default value, at any point in the path
    :return: obj.attr1.attr2.etc
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *default)

    attrs = dunder_delim_path.split(DUNDER)
    return functools.reduce(_getattr, [obj] + attrs)


def rhasattr(obj, dunder_delim_path: str):
    try:
        rgetattr(obj, dunder_delim_path)
        return True
    except AttributeError:
        return False

# https://stackoverflow.com/a/31174427/2683842
def rsetattr(obj, dunder_delim_path: str, val):
    """
    :param obj: Object
    :param dunder_delim_path: 'attr1__attr2__etc'
    :param val: obj.attr1.attr2.etc = val
    """
    parent, _, name = dunder_delim_path.rpartition(DUNDER)
    parent_obj = rgetattr(obj, parent) if parent else obj

    return setattr(parent_obj, name, val)
