import functools
import operator
from collections import defaultdict
from typing import *

import attr
from qtpy import QtWidgets as qw, QtCore as qc
from qtpy.QtCore import Slot
from qtpy.QtGui import QPalette, QColor, QFont
from qtpy.QtWidgets import QWidget

from corrscope.config import CorrError, DumpableAttrs, get_units
from corrscope.gui.util import color2hex
from corrscope.util import obj_name, perr
from corrscope.utils.trigger_util import lerp

if TYPE_CHECKING:
    from corrscope.gui import MainWindow
    from enum import Enum

    assert Enum

# TODO include all BoundWidget subclasses into this?
__all__ = [
    "PresentationModel",
    "map_gui",
    "behead",
    "rgetattr",
    "rsetattr",
    "Symbol",
    "SymbolText",
]


Signal = Any
WidgetUpdater = Callable[[], None]
Symbol = Hashable


SymbolText = Tuple[Symbol, str]


def _call_all(updaters: List[WidgetUpdater]):
    for updater in updaters:
        updater()


# Data binding presentation-model
class PresentationModel(qc.QObject):
    """Key-value MVP presentation-model.

    Qt's built-in model-view framework expects all models to
    take the form of a database-style numbered [row, column] structure,
    whereas my model takes the form of a key-value struct exposed as a form.

    Each GUI BoundWidget generally reads/writes `widget.pmodel[widget.path]`.

    To access cfg.foo.bar, set BoundWidget's path to read/write
    pmodel["foo.bar"] or pmodel["foo__bar"].

    To create a GUI field not directly mapping to `cfg`,
    define a property named "foo__baz", then set BoundWidget's path to read/write
    pmodel["foo.baz"] or pmodel["foo__baz"].
    """

    # These fields are specific to each subclass, and assigned there.
    # Although less explicit, these can be assigned using __init_subclass__.
    combo_symbol_text: Dict[str, Sequence[SymbolText]]
    edited = qc.Signal()

    def __init__(self, cfg: DumpableAttrs):
        super().__init__()
        self.cfg = cfg
        self.update_widget: Dict[str, List[WidgetUpdater]] = defaultdict(list)

    def __getitem__(self, item: str) -> Any:
        try:
            # Custom properties
            return getattr(self, item)
        except AttributeError:
            return rgetattr(self.cfg, item)

    def __setitem__(self, key, value):
        self.edited.emit()

        # Custom properties
        if hasattr(type(self), key):
            setattr(self, key, value)
        elif rhasattr(self.cfg, key):
            rsetattr(self.cfg, key, value)
        else:
            raise AttributeError(f"cannot set attribute {key} on {obj_name(self)}()")

    def set_cfg(self, cfg: DumpableAttrs):
        self.cfg = cfg
        for updater_list in self.update_widget.values():
            _call_all(updater_list)

    def update_all_bound(self, key: str):
        _call_all(self.update_widget[key])


SKIP_BINDING = "skip"


def map_gui(view: "MainWindow", model: PresentationModel) -> None:
    """
    Binding:
    - .ui <widget name="layout__nrows">
    - view.layout__nrows
    - pmodel['layout__nrows']

    Only <widget>s subclassing BoundWidget will be bound.
    """

    # dear pyqt, add generic mypy return types
    widgets: List[BoundWidget] = view.findChildren(BoundWidget)
    for widget in widgets:
        path = widget.objectName()

        # Exclude nameless ColorText inside BoundColorWidget wrapper,
        # since bind_widget(path="") will crash.
        # BoundColorWidget.bind_widget() handles binding children.
        if path != SKIP_BINDING:
            assert path != ""
            widget.bind_widget(model, path)


# Bound widgets
class BoundWidget(QWidget):
    default_palette: QPalette
    error_palette: QPalette

    pmodel: PresentationModel
    path: str

    def bind_widget(
        self, model: PresentationModel, path: str, connect_to_model=True
    ) -> None:
        """
        connect_to_model=False means:
        - self: ColorText is created and owned by BoundColorWidget wrapper.
        - Wrapper forwards model changes to self (which updates other widgets).
            (model.update_widget[path] != self)
        - self.gui_changed invokes self.set_model() (which updates model
            AND other widgets).
        - wrapper.gui_changed signal is NEVER emitted.
        """
        try:
            self.default_palette = self.palette()
            self.error_palette = self.calc_error_palette()

            self.pmodel = model
            self.path = path
            self.cfg2gui()

            if connect_to_model:
                # Allow widget to be updated by other events.
                model.update_widget[path].append(self.cfg2gui)

            # Allow pmodel to be changed by widget.
            self.gui_changed.connect(self.set_model)

        except Exception:
            perr(self)
            perr(path)
            raise

    # TODO unbind_widget(), model.update_widget[path].remove(self.cfg2gui)?

    def calc_error_palette(self) -> QPalette:
        """Palette with red background, used for widgets with invalid input."""
        error_palette = QPalette(self.palette())

        bg = error_palette.color(QPalette.Base)
        red = QColor(qc.Qt.red)

        red_bg = blend_colors(bg, red, 0.5)
        error_palette.setColor(QPalette.Base, red_bg)
        return error_palette

    # My class/method naming scheme is inconsistent and unintuitive.
    # PresentationModel+set_model vs. cfg2gui+set_gui vs. widget
    # Feel free to improve the naming.

    def cfg2gui(self) -> None:
        """Update the widget without triggering signals.

        When the presentation pmodel updates dependent widget 1,
        the pmodel (not widget 1) is responsible for updating other
        dependent widgets.
        TODO add option to send signals
        """
        with qc.QSignalBlocker(self):
            self.set_gui(self.pmodel[self.path])

    def set_gui(self, value):
        pass

    gui_changed: ClassVar[Signal]

    def set_model(self, value):
        for updater in self.pmodel.update_widget[self.path]:
            if updater != self.cfg2gui:
                updater()


def blend_colors(
    color1: QColor, color2: QColor, ratio: float, gamma: float = 2
) -> QColor:
    """Blends two colors in linear color space.
    Produces better results on both light and dark themes,
    than integer blending (which is too dark).
    """
    rgb1 = color1.getRgbF()[:3]  # r,g,b, remove alpha
    rgb2 = color2.getRgbF()[:3]
    rgb_blend = []

    for ch1, ch2 in zip(rgb1, rgb2):
        blend = lerp(ch1**gamma, ch2**gamma, ratio) ** (1 / gamma)
        rgb_blend.append(blend)

    return QColor.fromRgbF(*rgb_blend, 1.0)


# Mypy expects -> Callable[[BoundWidget, Any], None].
# PyCharm expects -> Callable[[Any], None].
# I give up.
def model_setter(value_type: type) -> Callable[..., None]:
    @Slot(value_type)
    def set_model(self: BoundWidget, value):
        assert isinstance(value, value_type)
        try:
            self.pmodel[self.path] = value
        except CorrError:
            self.setPalette(self.error_palette)
        else:
            BoundWidget.set_model(self, value)
            self.setPalette(self.default_palette)

    return set_model


def alias(name: str) -> property:
    return property(operator.attrgetter(name))


class BoundLineEdit(qw.QLineEdit, BoundWidget):
    # PyQt complains when we assign unbound methods (`set_gui = qw.QLineEdit.setText`),
    # but not if we call them indirectly.
    set_gui = alias("setText")
    gui_changed = alias("textChanged")
    set_model = model_setter(str)


class BoundSpinBox(qw.QSpinBox, BoundWidget):
    def bind_widget(self, model: PresentationModel, path: str, *args, **kwargs) -> None:
        BoundWidget.bind_widget(self, model, path, *args, **kwargs)
        try:
            parent, name = flatten_attr(model.cfg, path)
        except AttributeError:
            return

        fields = attr.fields_dict(type(parent))
        field = fields[name]
        self.setSuffix(get_units(field))

    set_gui = alias("setValue")
    gui_changed = alias("valueChanged")
    set_model = model_setter(int)


class BoundDoubleSpinBox(qw.QDoubleSpinBox, BoundWidget):
    bind_widget = BoundSpinBox.bind_widget

    set_gui = alias("setValue")
    gui_changed = alias("valueChanged")
    set_model = model_setter(float)


# CheckState inherits from int on PyQt5 and Enum on PyQt6. To compare integers with
# CheckState on both PyQt5 and 6, we have to call CheckState(int).
CheckState = qc.Qt.CheckState


class BoundCheckBox(qw.QCheckBox, BoundWidget):
    # setChecked accepts (bool).
    # setCheckState accepts (Qt.CheckState). Don't use it.
    set_gui = alias("setChecked")

    # stateChanged passes (Qt.CheckState).
    gui_changed = alias("stateChanged")

    # gui_changed -> set_model(Qt.CheckState).
    @Slot(int)
    def set_model(self, value: int):
        """Qt.CheckState.PartiallyChecked probably should not happen."""
        value = CheckState(value)
        assert value in [
            CheckState.Unchecked,
            CheckState.PartiallyChecked,
            CheckState.Checked,
        ]
        self.set_bool(value != CheckState.Unchecked)

    set_bool = model_setter(bool)


class BoundComboBox(qw.QComboBox, BoundWidget):
    """Combo box using values as immutable symbols.
    - Converts immutable values (from model) into symbols: value
    - Converts symbols (hard-coded) into immutable values: value
    """

    combo_symbol_text: Sequence[SymbolText]
    symbol2idx: Dict[Symbol, int]

    Custom = object()
    custom_if_unmatched: bool

    # noinspection PyAttributeOutsideInit
    def bind_widget(self, model: PresentationModel, path: str, *args, **kwargs) -> None:
        # Effectively enum values.
        self.combo_symbol_text = model.combo_symbol_text[path]

        # symbol2idx[enum] = combo-box index
        self.symbol2idx = {}
        self.custom_if_unmatched = False

        for i, (symbol, text) in enumerate(self.combo_symbol_text):
            self.symbol2idx[symbol] = i
            if symbol is self.Custom:
                self.custom_if_unmatched = True

            # Pretty-printed text
            self.addItem(text)

        BoundWidget.bind_widget(self, model, path, *args, **kwargs)

    # combobox.index = pmodel.attr
    def set_gui(self, model_value: Any) -> None:
        symbol = self._symbol_from_value(model_value)
        if self.custom_if_unmatched and symbol not in self.symbol2idx:
            combo_index = self.symbol2idx[self.Custom]
        else:
            combo_index = self.symbol2idx[self._symbol_from_value(model_value)]

        self.setCurrentIndex(combo_index)

    @staticmethod
    def _symbol_from_value(value) -> Symbol:
        return value

    gui_changed = alias("currentIndexChanged")

    # pmodel.attr = combobox.index
    @Slot(int)
    def set_model(self, combo_index: int):
        assert isinstance(combo_index, int)
        combo_symbol, _ = self.combo_symbol_text[combo_index]
        if combo_symbol is not self.Custom:
            self.pmodel[self.path] = self._value_from_symbol(combo_symbol)
            BoundWidget.set_model(self, None)

    @staticmethod
    def _value_from_symbol(symbol: Symbol):
        return symbol


class TypeComboBox(BoundComboBox):
    """Combo box using types as immutable symbols.
    - Converts mutable values (from model) into symbols: type(value)
    - Converts symbols (hard-coded) into mutable values: cls()
    """

    @staticmethod
    def _symbol_from_value(instance) -> type:
        return type(instance)

    @staticmethod
    def _value_from_symbol(obj_type: type):
        return obj_type()


def _format_font_size(size: float) -> str:
    """Strips away trailing .0 from 13.0.
    Basically unused, since QFontDialog will only allow integer font sizes."""
    return ("%f" % size).rstrip("0").rstrip(".")


class BoundFontButton(qw.QPushButton, BoundWidget):
    def __init__(self, parent: qw.QWidget):
        qw.QPushButton.__init__(self, parent)
        self.clicked.connect(self.on_clicked)

    def set_gui(self, qfont: QFont):
        self.setText(qfont.family() + " " + _format_font_size(qfont.pointSizeF()))

        preview_font = QFont(qfont)
        preview_font.setPointSizeF(self.font().pointSizeF())
        self.setFont(preview_font)

    @Slot()
    def on_clicked(self):
        old_font: QFont = self.pmodel[self.path]

        # https://doc.qt.io/qtforpython/PySide2/QtWidgets/QFontDialog.html#detailed-description
        # is wrong.
        (new_font, ok) = qw.QFontDialog.getFont(old_font, self.window())
        if ok:
            self.set_gui(new_font)
            self.gui_changed.emit(new_font)

    gui_changed = qc.Signal(QFont)

    set_model = model_setter(QFont)


# Color-specific widgets


class BoundColorWidget(BoundWidget, qw.QWidget):
    """
    - set_gui(): Model is sent to self.text, which updates all other widgets.
    - self.text: ColorText
    - When self.text changes, it converts to hex, then updates the pmodel
        and sends signal `hex_color` which updates check/button.
    - self does NOT update the pmodel. (gui_changed is never emitted.)
    """

    optional = False

    def __init__(self, parent: qw.QWidget):
        qw.QWidget.__init__(self, parent)

        layout = qw.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Setup text field.
        self.text = _ColorText(self, self.optional)
        layout.addWidget(self.text)  # http://doc.qt.io/qt-5/qlayout.html#addItem

        # Setup checkbox
        if self.optional:
            self.check = _ColorCheckBox(self, self.text)
            self.check.setToolTip("Enable/Disable Color")
            layout.addWidget(self.check)

        # Setup colored button.
        self.button = _ColorButton(self, self.text)
        self.button.setToolTip("Pick Color")
        layout.addWidget(self.button)

    # override BoundWidget
    def bind_widget(self, model: PresentationModel, path: str, *args, **kwargs) -> None:
        super().bind_widget(model, path, *args, **kwargs)
        self.text.bind_widget(model, path, connect_to_model=False)

    # impl BoundWidget
    def set_gui(self, value: Optional[str]):
        self.text.set_gui(value)

    # impl BoundWidget
    # Never gets emitted. self.text.set_model is responsible for updating model.
    gui_changed = qc.Signal(str)

    # impl BoundWidget
    # Never gets called.
    def set_model(self, value):
        raise RuntimeError(
            "BoundColorWidget.gui_changed -> set_model should not be called"
        )


class OptionalColorWidget(BoundColorWidget):
    optional = True


class _ColorText(BoundLineEdit):
    """
    - Validates and converts colors to hex (from model AND gui)
    - If __init__ optional, special-cases missing colors.
    """

    def __init__(self, parent: QWidget, optional: bool):
        super().__init__(parent)
        self.setObjectName(SKIP_BINDING)
        self.optional = optional

    hex_color = qc.Signal(str)

    def set_gui(self, value: Optional[str]):
        """model2gui"""

        if self.optional and not value:
            value = ""
        else:
            value = color2hex(value)  # raises CorrError if invalid.

        # Don't write back to model immediately.
        # Loading is a const process, only editing the GUI should change the model.
        with qc.QSignalBlocker(self):
            self.setText(value)

        # Write to other GUI widgets immediately.
        self.hex_color.emit(value)  # calls button.set_color()

    @Slot(str)
    def set_model(self: BoundWidget, value: str):
        """gui2model"""

        if self.optional and not value:
            value = None
        else:
            try:
                value = color2hex(value)
            except CorrError:
                self.setPalette(self.error_palette)
                self.hex_color.emit("")  # calls button.set_color()
                return

        self.setPalette(self.default_palette)
        self.hex_color.emit(value or "")  # calls button.set_color()
        self.pmodel[self.path] = value
        BoundWidget.set_model(self, value)

    def sizeHint(self) -> qc.QSize:
        """Reduce the width taken up by #rrggbb color text boxes."""
        return self.minimumSizeHint()


class _ColorButton(qw.QPushButton):
    def __init__(self, parent: QWidget, text: "_ColorText"):
        qw.QPushButton.__init__(self, parent)
        self.clicked.connect(self.on_clicked)

        # Initialize "current color"
        self.curr_color = QColor()

        # Initialize text
        self.color_text = text
        text.hex_color.connect(self.set_color)

    @Slot()
    def on_clicked(self):
        # https://bugreports.qt.io/browse/QTBUG-38537
        # On Windows, QSpinBox height is wrong if stylesheets are enabled.
        # And QColorDialog(parent=self) contains QSpinBox and inherits our stylesheets.
        # So set parent=self.window().

        color: QColor = qw.QColorDialog.getColor(self.curr_color, self.window())
        if not color.isValid():
            return

        self.color_text.setText(color.name())  # textChanged calls self.set_color()

    @Slot(str)
    def set_color(self, hex_color: str):
        color = QColor(hex_color)
        self.curr_color = color

        if color.isValid():
            # Tooltips inherit our styles. Don't change their background.
            qss = f"{obj_name(self)} {{ background: {color.name()}; }}"
        else:
            qss = ""

        self.setStyleSheet(qss)


class _ColorCheckBox(qw.QCheckBox):
    def __init__(self, parent: QWidget, text: "_ColorText"):
        qw.QCheckBox.__init__(self, parent)
        self.stateChanged.connect(self.on_check)

        self.color_text = text
        text.hex_color.connect(self.set_color)

    @Slot(str)
    def set_color(self, hex_color: str):
        with qc.QSignalBlocker(self):
            self.setChecked(bool(hex_color))

    @Slot(int)
    def on_check(self, value: int):
        """Qt.CheckState.PartiallyChecked probably should not happen."""
        value = CheckState(value)
        assert value in [
            CheckState.Unchecked,
            CheckState.PartiallyChecked,
            CheckState.Checked,
        ]
        if value != CheckState.Unchecked:
            self.color_text.setText("#ffffff")
        else:
            self.color_text.setText("")


# Unused
def try_behead(string: str, header: str) -> Optional[str]:
    if not string.startswith(header):
        return None
    return string[len(header) :]


def behead(string: str, header: str) -> str:
    if not string.startswith(header):
        raise ValueError(f"{string} does not start with {header}")
    return string[len(header) :]


SEPARATOR = "."


def strip_dunders(dunder_delim_path: str):
    return dunder_delim_path.replace("__", SEPARATOR)


# https://gist.github.com/wonderbeyond/d293e7a2af1de4873f2d757edd580288
def rgetattr(obj: DumpableAttrs, dunder_delim_path: str, *default) -> Any:
    """
    :param obj: Object
    :param dunder_delim_path: 'attr1__attr2.etc' (__ and . are equivalent)
    :param default: Optional default value, at any point in the path
    :return: obj.attr1.attr2.etc
    """

    path = strip_dunders(dunder_delim_path)
    attrs: List[Any] = path.split(SEPARATOR)
    try:
        return functools.reduce(getattr, attrs, obj)
    except AttributeError:
        if default:
            return default[0]
        raise


def rhasattr(obj, dunder_delim_path: str):
    try:
        rgetattr(obj, dunder_delim_path)
        return True
    except AttributeError:
        return False


def flatten_attr(obj, dunder_delim_path: str) -> Tuple[Any, str]:
    """
    :param obj: Object
    :param dunder_delim_path: 'attr1__attr2.etc' (__ and . are equivalent)
    :return: (shallow_obj, name) such that
        getattr(shallow_obj, name) == rgetattr(obj, dunder_delim_path).
    """

    path = strip_dunders(dunder_delim_path)
    parent, _, name = path.rpartition(SEPARATOR)
    parent_obj = rgetattr(obj, parent) if parent else obj

    return parent_obj, name


# https://stackoverflow.com/a/31174427/2683842
def rsetattr(obj, dunder_delim_path: str, val):
    """
    :param obj: Object
    :param dunder_delim_path: 'attr1__attr2.etc' (__ and . are equivalent)
    :param val: obj.attr1.attr2.etc = val
    """
    parent_obj, name = flatten_attr(obj, dunder_delim_path)
    return setattr(parent_obj, name, val)
