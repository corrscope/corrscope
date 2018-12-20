import functools
import operator
from typing import Optional, List, Callable, Dict, Any, ClassVar, TYPE_CHECKING

from PyQt5 import QtWidgets as qw, QtCore as qc
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QWidget

from corrscope.config import CorrError
from corrscope.triggers import lerp
from corrscope.util import obj_name, perr

if TYPE_CHECKING:
    from corrscope.gui import MainWindow

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
        if hasattr(type(self), key):
            setattr(self, key, value)
        elif rhasattr(self.cfg, key):
            rsetattr(self.cfg, key, value)
        else:
            raise AttributeError(f'cannot set attribute {key} on {obj_name(self)}()')

    def set_cfg(self, cfg: Attrs):
        self.cfg = cfg
        for updater in self.update_widget.values():
            updater()


# TODO add tests for recursive operations
def map_gui(view: 'MainWindow', model: PresentationModel):
    """
    Binding:
    - .ui <widget name="layout__nrows">
    - view.layout__nrows
    - pmodel['layout__nrows']

    Only <widget>s subclassing BoundWidget will be bound.
    """

    widgets: List[BoundWidget] = view.findChildren(BoundWidget)  # dear pyqt, add generic mypy return types
    for widget in widgets:
        path = widget.objectName()
        widget.bind_widget(model, path)
        widget.gui_changed.connect(view.on_gui_edited)


Signal = Any

class BoundWidget(QWidget):
    default_palette: QPalette
    error_palette: QPalette

    pmodel: PresentationModel
    path: str

    def bind_widget(self, model: PresentationModel, path: str) -> None:
        try:
            self.default_palette = self.palette()
            self.error_palette = self.calc_error_palette()

            self.pmodel = model
            self.path = path
            self.cfg2gui()

            # Allow widget to be updated by other events.
            model.update_widget[path] = self.cfg2gui

            # Allow pmodel to be changed by widget.
            self.gui_changed.connect(self.set_model)

        except Exception:
            perr(self)
            perr(path)
            raise

    def calc_error_palette(self) -> QPalette:
        """ Palette with red background, used for widgets with invalid input. """
        error_palette = QPalette(self.palette())

        bg = error_palette.color(QPalette.Base)
        red = QColor(qc.Qt.red)

        red_bg = blend_colors(bg, red, 0.5)
        error_palette.setColor(QPalette.Base, red_bg)
        return error_palette

    def cfg2gui(self):
        """ Update the widget without triggering signals.

        When the presentation pmodel updates dependent widget 1,
        the pmodel (not widget 1) is responsible for updating other
        dependent widgets.
        TODO add option to send signals
        """
        with qc.QSignalBlocker(self):
            self.set_gui(self.pmodel[self.path])

    def set_gui(self, value): pass

    gui_changed: ClassVar[Signal]

    def set_model(self, value): pass


def blend_colors(color1: QColor, color2: QColor, ratio: float, gamma=2):
    """ Blends two colors in linear color space.
    Produces better results on both light and dark themes,
    than integer blending (which is too dark).
    """
    rgb1 = color1.getRgbF()[:3]  # r,g,b, remove alpha
    rgb2 = color2.getRgbF()[:3]
    rgb_blend = []

    for ch1, ch2 in zip(rgb1, rgb2):
        blend = lerp(ch1 ** gamma, ch2 ** gamma, ratio) ** (1/gamma)
        rgb_blend.append(blend)

    return QColor.fromRgbF(*rgb_blend, 1.0)


def model_setter(value_type: type) -> Callable:
    @pyqtSlot(value_type)
    def set_model(self: BoundWidget, value):
        assert isinstance(value, value_type)
        try:
            self.pmodel[self.path] = value
        except CorrError:
            self.setPalette(self.error_palette)
        else:
            self.setPalette(self.default_palette)
    return set_model


def alias(name: str):
    return property(operator.attrgetter(name))


class BoundLineEdit(qw.QLineEdit, BoundWidget):
    # PyQt complains when we assign unbound methods (`set_gui = qw.QLineEdit.setText`),
    # but not if we call them indirectly.
    set_gui = alias('setText')
    gui_changed = alias('textChanged')
    set_model = model_setter(str)


class BoundSpinBox(qw.QSpinBox, BoundWidget):
    set_gui = alias('setValue')
    gui_changed = alias('valueChanged')
    set_model = model_setter(int)


class BoundDoubleSpinBox(qw.QDoubleSpinBox, BoundWidget):
    set_gui = alias('setValue')
    gui_changed = alias('valueChanged')
    set_model = model_setter(float)


class BoundComboBox(qw.QComboBox, BoundWidget):
    combo_symbols: List[str]
    symbol2idx: Dict[str, int]

    # noinspection PyAttributeOutsideInit
    def bind_widget(self, model: PresentationModel, path: str) -> None:
        # Effectively enum values.
        self.combo_symbols = model.combo_symbols[path]

        # symbol2idx[str] = int
        self.symbol2idx = {}

        # Pretty-printed text
        combo_text = model.combo_text[path]
        for i, symbol in enumerate(self.combo_symbols):
            self.symbol2idx[symbol] = i
            self.addItem(combo_text[i])

        BoundWidget.bind_widget(self, model, path)

    # combobox.index = pmodel.attr
    def set_gui(self, symbol: str):
        combo_index = self.symbol2idx[symbol]
        self.setCurrentIndex(combo_index)

    gui_changed = alias('currentIndexChanged')

    # pmodel.attr = combobox.index
    @pyqtSlot(int)
    def set_model(self, combo_index: int):
        assert isinstance(combo_index, int)
        self.pmodel[self.path] = self.combo_symbols[combo_index]


# Unused
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
