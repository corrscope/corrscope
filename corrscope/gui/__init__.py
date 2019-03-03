import functools
import sys
import traceback
from collections import OrderedDict
from pathlib import Path
from types import MethodType
from typing import Optional, List, Any, Tuple, Callable, Union, Dict, Sequence

import PyQt5.QtCore as qc
import PyQt5.QtWidgets as qw
import attr
from PyQt5 import uic
from PyQt5.QtCore import QModelIndex, Qt
from PyQt5.QtCore import QVariant
from PyQt5.QtGui import QKeySequence, QFont, QCloseEvent
from PyQt5.QtWidgets import QShortcut

import corrscope
import corrscope.settings.global_prefs as gp
from corrscope import cli
from corrscope.channel import ChannelConfig
from corrscope.config import CorrError, copy_config, yaml
from corrscope.corrscope import CorrScope, Config, Arguments, default_config
from corrscope.gui.data_bind import (
    PresentationModel,
    map_gui,
    behead,
    rgetattr,
    rsetattr,
    Symbol,
)
from corrscope.gui.history_file_dlg import (
    get_open_file_name,
    get_open_file_list,
    get_save_file_path,
)
from corrscope.gui.util import color2hex, Locked, find_ranges, TracebackDialog
from corrscope.layout import Orientation, StereoOrientation
from corrscope.outputs import IOutputConfig, FFplayOutputConfig, FFmpegOutputConfig
from corrscope.settings import paths
from corrscope.triggers import CorrelationTriggerConfig, ITriggerConfig, SpectrumConfig
from corrscope.util import obj_name
from corrscope.wave import Flatten

FILTER_WAV_FILES = ["WAV files (*.wav)"]

APP_NAME = f"{corrscope.app_name} {corrscope.__version__}"
APP_DIR = Path(__file__).parent


def res(file: str) -> str:
    return str(APP_DIR / file)


def gui_main(cfg_or_path: Union[Config, Path]):
    # qw.QApplication.setStyle('fusion')
    QApp = qw.QApplication
    QApp.setAttribute(qc.Qt.AA_EnableHighDpiScaling)

    # Qt on Windows will finally switch default font to lfMessageFont=Segoe UI
    # (Vista, 2006)... in 2020 (Qt 6.0).
    if qc.QSysInfo.kernelType() == "winnt":
        # This will be wrong for non-English languages, but it's better than default?
        font = QFont("Segoe UI", 9)
        font.setStyleHint(QFont.SansSerif)
        QApp.setFont(font)

    app = qw.QApplication(sys.argv)
    window = MainWindow(cfg_or_path)
    sys.exit(app.exec_())


class MainWindow(qw.QMainWindow):
    """
    Main window.

    Control flow:
    __init__: either
    - load_cfg
    - load_cfg_from_path

    Opening a document:
    - load_cfg_from_path
    """

    def __init__(self, cfg_or_path: Union[Config, Path]):
        super().__init__()

        # Load settings.
        self.pref = gp.load_prefs()

        # Load UI.
        uic.loadUi(res("mainwindow.ui"), self)  # sets windowTitle

        # Bind UI buttons, etc. Functions block main thread, avoiding race conditions.
        self.master_audio_browse.clicked.connect(self.on_master_audio_browse)

        self.channelUp.add_shortcut(self.channelsGroup, "ctrl+shift+up")
        self.channelDown.add_shortcut(self.channelsGroup, "ctrl+shift+down")

        self.channelUp.clicked.connect(self.channel_view.on_channel_up)
        self.channelDown.clicked.connect(self.channel_view.on_channel_down)
        self.channelAdd.clicked.connect(self.on_channel_add)
        self.channelDelete.clicked.connect(self.on_channel_delete)

        # Bind actions.
        self.action_separate_render_dir.setChecked(self.pref.separate_render_dir)
        self.action_separate_render_dir.toggled.connect(
            self.on_separate_render_dir_toggled
        )

        self.actionNew.triggered.connect(self.on_action_new)
        self.actionOpen.triggered.connect(self.on_action_open)
        self.actionSave.triggered.connect(self.on_action_save)
        self.actionSaveAs.triggered.connect(self.on_action_save_as)
        self.actionPreview.triggered.connect(self.on_action_preview)
        self.actionRender.triggered.connect(self.on_action_render)
        self.actionExit.triggered.connect(qw.QApplication.closeAllWindows)

        # Initialize CorrScope-thread attribute.
        self.corr_thread: Optional[CorrThread] = None

        # Bind config to UI.
        if isinstance(cfg_or_path, Config):
            self.load_cfg(cfg_or_path, None)
        elif isinstance(cfg_or_path, Path):
            self.load_cfg_from_path(cfg_or_path)
        else:
            raise TypeError(
                f"argument cfg={cfg_or_path} has invalid type {obj_name(cfg_or_path)}"
            )

        self.show()

    _cfg_path: Optional[Path]

    # Whether document is dirty, changed, has unsaved changes
    _any_unsaved: bool

    @property
    def any_unsaved(self) -> bool:
        return self._any_unsaved

    @any_unsaved.setter
    def any_unsaved(self, value: bool):
        self._any_unsaved = value
        self._update_unsaved_title()

    # GUI layout widgets
    tabWidget: qw.QTabWidget

    # Config models
    model: Optional["ConfigModel"] = None

    channel_model: "ChannelModel"
    channel_view: "ChannelTableView"
    channelsGroup: qw.QGroupBox

    def prompt_save(self) -> bool:
        """
        Called when user is closing document
        (when opening a new document or closing the app).

        :return: False if user cancels close-document action.
        """
        if not self.any_unsaved:
            return True

        Msg = qw.QMessageBox

        save_message = f"Save changes to {self.title_cache}?"
        should_close = Msg.question(
            self, "Save Changes?", save_message, Msg.Save | Msg.Discard | Msg.Cancel
        )

        if should_close == Msg.Cancel:
            return False
        elif should_close == Msg.Discard:
            return True
        else:
            return self.on_action_save()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Called on closing window."""
        if self.prompt_save():
            gp.dump_prefs(self.pref)
            event.accept()
        else:
            event.ignore()

    def on_action_new(self):
        if not self.prompt_save():
            return
        cfg = default_config()
        self.load_cfg(cfg, None)

    def on_action_open(self):
        if not self.prompt_save():
            return
        name = get_open_file_name(
            self.pref.file_dir_ref, self, "Open config", ["YAML files (*.yaml)"]
        )
        if name:
            cfg_path = Path(name)
            self.load_cfg_from_path(cfg_path)

    def load_cfg_from_path(self, cfg_path: Path):
        # Bind GUI to dummy config, in case loading cfg_path raises Exception.
        if self.model is None:
            self.load_cfg(default_config(), None)

        assert cfg_path.is_file()
        self.pref.file_dir = str(cfg_path.parent.resolve())

        try:
            # Raises YAML structural exceptions
            cfg = yaml.load(cfg_path)

            # Raises color getter exceptions
            # FIXME if error halfway, clear "file path" and load empty model.
            self.load_cfg(cfg, cfg_path)

        except Exception as e:
            qw.QMessageBox.critical(self, "Error loading file", str(e))
            return

    def load_cfg(self, cfg: Config, cfg_path: Optional[Path]) -> None:
        self._cfg_path = cfg_path
        self._any_unsaved = False
        self.load_title()
        self.tabWidget.setCurrentIndex(0)

        if self.model is None:
            self.model = ConfigModel(cfg)
            self.model.edited.connect(self.on_model_edited)
            # Calls self.on_gui_edited() whenever GUI widgets change.
            map_gui(self, self.model)
        else:
            self.model.set_cfg(cfg)

        self.channel_model = ChannelModel(cfg.channels)
        # Calling setModel again disconnects previous model.
        self.channel_view.setModel(self.channel_model)
        self.channel_model.dataChanged.connect(self.on_model_edited)
        self.channel_model.rowsInserted.connect(self.on_model_edited)
        self.channel_model.rowsMoved.connect(self.on_model_edited)
        self.channel_model.rowsRemoved.connect(self.on_model_edited)

    def on_model_edited(self):
        self.any_unsaved = True

    title_cache: str

    def load_title(self) -> None:
        self.title_cache = self.title
        self._update_unsaved_title()

    def _update_unsaved_title(self) -> None:
        if self.any_unsaved:
            undo_str = "*"
        else:
            undo_str = ""
        self.setWindowTitle(f"{self.title_cache}{undo_str} - {APP_NAME}")

    # GUI actions, etc.
    master_audio_browse: qw.QPushButton
    channelAdd: "ShortcutButton"
    channelDelete: "ShortcutButton"
    channelUp: "ShortcutButton"
    channelDown: "ShortcutButton"

    action_separate_render_dir: qw.QAction
    # Loading mainwindow.ui changes menuBar from a getter to an attribute.
    menuBar: qw.QMenuBar
    actionNew: qw.QAction
    actionOpen: qw.QAction
    actionSave: qw.QAction
    actionSaveAs: qw.QAction
    actionPreview: qw.QAction
    actionRender: qw.QAction
    actionExit: qw.QAction

    def on_master_audio_browse(self):
        name = get_open_file_name(
            self.pref.file_dir_ref, self, "Open master audio file", FILTER_WAV_FILES
        )
        if name:
            master_audio = "master_audio"
            self.model[master_audio] = name
            self.model.update_widget[master_audio]()

    def on_separate_render_dir_toggled(self, checked: bool):
        self.pref.separate_render_dir = checked
        if checked:
            self.pref.render_dir = self.pref.file_dir
        else:
            self.pref.render_dir = ""

    def on_channel_add(self):
        wavs = get_open_file_list(
            self.pref.file_dir_ref, self, "Add audio channels", FILTER_WAV_FILES
        )
        if wavs:
            self.channel_view.append_channels(wavs)

    def on_channel_delete(self):
        self.channel_view.delete_selected()

    def on_action_save(self) -> bool:
        """
        :return: False if user cancels save action.
        """
        if self._cfg_path is None:
            return self.on_action_save_as()

        yaml.dump(self.cfg, self._cfg_path)
        self.any_unsaved = False
        self._update_unsaved_title()
        return True

    def on_action_save_as(self) -> bool:
        """
        :return: False if user cancels save action.
        """
        cfg_path_default = self.file_stem + cli.YAML_NAME

        filters = ["YAML files (*.yaml)", "All files (*)"]
        path = get_save_file_path(
            self.pref.file_dir_ref,
            self,
            "Save As",
            cfg_path_default,
            filters,
            cli.YAML_NAME,
        )
        if path:
            self._cfg_path = path
            self.load_title()
            self.on_action_save()
            return True
        else:
            return False

    def on_action_preview(self):
        """ Launch CorrScope and ffplay. """
        error_msg = "Cannot play, another play/render is active"
        if self.corr_thread is not None:
            qw.QMessageBox.critical(self, "Error", error_msg)
            return

        outputs = [FFplayOutputConfig()]
        self.play_thread(outputs, dlg=None)

    def on_action_render(self):
        """ Get file name. Then show a progress dialog while rendering to file. """
        error_msg = "Cannot render to file, another play/render is active"
        if self.corr_thread is not None:
            qw.QMessageBox.critical(self, "Error", error_msg)
            return

        video_path = self.file_stem + cli.VIDEO_NAME
        filters = ["MP4 files (*.mp4)", "All files (*)"]

        # Points to either `file_dir` or `render_dir`.
        ref = self.pref.render_dir_ref

        path = get_save_file_path(
            ref, self, "Render to Video", video_path, filters, cli.VIDEO_NAME
        )
        if path:
            name = str(path)
            dlg = CorrProgressDialog(self, "Rendering video")

            outputs = [FFmpegOutputConfig(name)]
            self.play_thread(outputs, dlg)

    def play_thread(
        self, outputs: List[IOutputConfig], dlg: Optional["CorrProgressDialog"]
    ):
        assert self.model

        arg = self._get_args(outputs)
        cfg = copy_config(self.model.cfg)
        t = self.corr_thread = CorrThread(cfg, arg)

        if dlg:
            dlg.canceled.connect(t.abort)
            t.arg = attr.evolve(
                arg,
                on_begin=run_on_ui_thread(dlg.on_begin, (float, float)),
                progress=run_on_ui_thread(dlg.setValue, (int,)),
                is_aborted=t.is_aborted.get,
                on_end=run_on_ui_thread(dlg.reset, ()),  # TODO dlg.close
            )

        t.finished.connect(self.on_play_thread_finished)
        t.error.connect(self.on_play_thread_error)
        t.ffmpeg_missing.connect(self.on_play_thread_ffmpeg_missing)
        t.start()

    def _get_args(self, outputs: List[IOutputConfig]):
        arg = Arguments(cfg_dir=self.cfg_dir, outputs=outputs)
        return arg

    def on_play_thread_finished(self):
        self.corr_thread = None

    def on_play_thread_error(self, stack_trace: str):
        TracebackDialog(self).showMessage(stack_trace)

    def on_play_thread_ffmpeg_missing(self):
        DownloadFFmpegActivity(self)

    # File paths
    @property
    def cfg_dir(self) -> str:
        """Only used when generating Arguments when playing corrscope.
        Not used to determine default path of file dialogs."""
        maybe_path = self._cfg_path or self.cfg.master_audio
        if maybe_path:
            # Windows likes to raise OSError when path contains *
            try:
                return str(Path(maybe_path).resolve().parent)
            except OSError:
                return "."

        return "."

    UNTITLED = "Untitled"

    @property
    def title(self) -> str:
        if self._cfg_path:
            return self._cfg_path.name
        return self.UNTITLED

    @property
    def file_stem(self) -> str:
        """ Returns a "config name" with no dots or slashes. """
        return cli.get_name(self._cfg_path or self.cfg.master_audio)

    @property
    def cfg(self):
        return self.model.cfg


class CorrThread(qc.QThread):
    is_aborted: Locked[bool]

    @qc.pyqtSlot()
    def abort(self):
        self.is_aborted.set(True)

    error = qc.pyqtSignal(str)
    ffmpeg_missing = qc.pyqtSignal()

    def __init__(self, cfg: Config, arg: Arguments):
        qc.QThread.__init__(self)
        self.cfg = cfg
        self.arg = arg
        self.is_aborted = Locked(False)

    def run(self) -> None:
        cfg = self.cfg
        arg = self.arg
        try:
            CorrScope(cfg, arg).play()

        except paths.MissingFFmpegError:
            arg.on_end()
            self.ffmpeg_missing.emit()

        except Exception as e:
            arg.on_end()
            if isinstance(e, CorrError):
                stack_trace = traceback.format_exc(limit=0)
            else:
                stack_trace = traceback.format_exc()
            self.error.emit(stack_trace)

        else:
            arg.on_end()


class CorrProgressDialog(qw.QProgressDialog):
    def __init__(self, parent: Optional[qw.QWidget], title: str):
        super().__init__(parent)
        self.setMinimumWidth(300)
        self.setWindowTitle(title)
        self.setLabelText("Progress:")

        # If set to 0, the dialog is always shown as soon as any progress is set.
        self.setMinimumDuration(0)

        # Don't reset when rendering is approximately finished.
        self.setAutoReset(False)

        # Close after CorrScope finishes.
        self.setAutoClose(True)

    @qc.pyqtSlot(float, float)
    def on_begin(self, begin_time, end_time):
        self.setRange(int(round(begin_time)), int(round(end_time)))
        # self.setValue is called by CorrScope, on the first frame.


# *arg_types: type
def run_on_ui_thread(
    bound_slot: MethodType, types: Tuple[type, ...]
) -> Callable[..., None]:
    """ Runs an object's slot on the object's own thread.
    It's terrible code but it works (as long as the slot has no return value).
    """
    qmo = qc.QMetaObject

    # QObject *obj,
    obj = bound_slot.__self__

    # const char *member,
    member = bound_slot.__name__

    # Qt::ConnectionType type,
    # QGenericReturnArgument ret,
    # https://riverbankcomputing.com/pipermail/pyqt/2014-December/035223.html
    conn = Qt.QueuedConnection

    @functools.wraps(bound_slot)
    def inner(*args):
        if len(types) != len(args):
            raise TypeError(f"len(types)={len(types)} != len(args)={len(args)}")

        # https://www.qtcentre.org/threads/29156-Calling-a-slot-from-another-thread?p=137140#post137140
        # QMetaObject.invokeMethod(skypeThread, "startSkypeCall", Qt.QueuedConnection, QtCore.Q_ARG("QString", "someguy"))

        _args = [qc.Q_ARG(typ, typ(arg)) for typ, arg in zip(types, args)]
        return qmo.invokeMethod(obj, member, conn, *_args)

    return inner


class ShortcutButton(qw.QPushButton):
    scoped_shortcut: QShortcut

    def add_shortcut(self, scope: qw.QWidget, shortcut: str) -> None:
        """ Adds shortcut and tooltip. """
        keys = QKeySequence(shortcut, QKeySequence.PortableText)

        self.scoped_shortcut = qw.QShortcut(keys, scope)
        self.scoped_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.scoped_shortcut.activated.connect(self.click)

        self.setToolTip(keys.toString(QKeySequence.NativeText))


# Begin ConfigModel properties


def nrow_ncol_property(altered: str, unaltered: str) -> property:
    def get(self: "ConfigModel"):
        val = getattr(self.cfg.layout, altered)
        if val is None:
            return 0
        else:
            return val

    def set(self: "ConfigModel", val: int):
        if val > 0:
            setattr(self.cfg.layout, altered, val)
            setattr(self.cfg.layout, unaltered, None)
            self.update_widget["layout__" + unaltered]()
        elif val == 0:
            setattr(self.cfg.layout, altered, None)
        else:
            raise CorrError(f"invalid input: {altered} < 0, should never happen")

    return property(get, set)


def default_property(path: str, default: Any) -> property:
    def getter(self: "ConfigModel"):
        val = rgetattr(self.cfg, path)
        if val is None:
            return default
        else:
            return val

    def setter(self: "ConfigModel", val):
        rsetattr(self.cfg, path, val)

    return property(getter, setter)


def path_strip_quotes(path: str) -> str:
    if len(path) and path[0] == path[-1] == '"':
        return path[1:-1]
    return path


def path_fix_property(path: str) -> property:
    """Removes quotes from paths, when setting from GUI."""

    def getter(self: "ConfigModel") -> str:
        return rgetattr(self.cfg, path)

    def setter(self: "ConfigModel", val: str):
        val = path_strip_quotes(val)
        rsetattr(self.cfg, path, val)

    return property(getter, setter)


flatten_no_stereo = {
    Flatten.SumAvg: "Average: (L+R)/2",
    Flatten.DiffAvg: "DiffAvg: (L-R)/2",
}
flatten_modes = {**flatten_no_stereo, Flatten.Stereo: "Stereo"}
assert set(flatten_modes.keys()) == set(Flatten.modes)  # type: ignore


class ConfigModel(PresentationModel):
    cfg: Config
    combo_symbols: Dict[str, Sequence[Symbol]] = {}
    combo_text: Dict[str, Sequence[str]] = {}

    master_audio = path_fix_property("master_audio")

    # Stereo flattening
    for path, symbol_map in [
        ["trigger_stereo", flatten_no_stereo],
        ["render_stereo", flatten_modes],
    ]:
        combo_symbols[path] = list(symbol_map.keys())
        combo_text[path] = list(symbol_map.values())
    del path, symbol_map

    @property
    def render_resolution(self) -> str:
        render = self.cfg.render
        w, h = render.width, render.height
        return f"{w}x{h}"

    @render_resolution.setter
    def render_resolution(self, value: str):
        error = CorrError(f"invalid resolution {value}, must be WxH")

        for sep in "x*,":
            width_height = value.split(sep)
            if len(width_height) == 2:
                break
        else:
            raise error

        render = self.cfg.render
        width, height = width_height
        try:
            render.width = int(width)
            render.height = int(height)
        except ValueError:
            raise error

    layout__nrows = nrow_ncol_property("nrows", unaltered="ncols")
    layout__ncols = nrow_ncol_property("ncols", unaltered="nrows")

    _orientations = [["h", "Horizontal"], ["v", "Vertical"]]
    _stereo_orientations = _orientations + [["overlay", "Overlay"]]

    for path, cls, symbol_map in [
        ["layout__orientation", Orientation, _orientations],
        ["layout__stereo_orientation", StereoOrientation, _stereo_orientations],
    ]:
        symbol_map = OrderedDict(symbol_map)
        # comprehensions fail in class scope
        combo_symbols[path] = list(map(cls, symbol_map.keys()))
        combo_text[path] = list(symbol_map.values())
    del path, cls, symbol_map

    render__line_width = default_property("render__line_width", 1.5)

    @property
    def trigger__pitch_tracking(self) -> bool:
        scfg = self.cfg.trigger.pitch_tracking
        gui = scfg is not None
        return gui

    @trigger__pitch_tracking.setter
    def trigger__pitch_tracking(self, gui: bool):
        scfg = SpectrumConfig() if gui else None
        self.cfg.trigger.pitch_tracking = scfg


# End ConfigModel


class ChannelTableView(qw.QTableView):
    def append_channels(self, wavs: List[str]):
        model: ChannelModel = self.model()

        begin_row = model.rowCount()
        count_rows = len(wavs)

        col = model.idx_of_key["wav_path"]

        model.insertRows(begin_row, count_rows)
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


@attr.dataclass
class Column:
    key: str
    cls: Union[type, Callable[[str], Any]]
    default: Any

    def _display_name(self) -> str:
        return self.key.replace("__", "\n").replace("_", " ").title()

    display_name: str = attr.Factory(_display_name, takes_self=True)


class ChannelModel(qc.QAbstractTableModel):
    """ Design based off
    https://doc.qt.io/qt-5/model-view-programming.html#a-read-only-example-model and
    https://doc.qt.io/qt-5/model-view-programming.html#model-subclassing-reference
    """

    def __init__(self, channels: List[ChannelConfig]):
        """ Mutates `channels` and `line_color` for convenience. """
        super().__init__()
        self.channels = channels

        line_color = "line_color"

        for cfg in self.channels:
            t = cfg.trigger
            if isinstance(t, ITriggerConfig):
                if not isinstance(t, CorrelationTriggerConfig):
                    raise CorrError(f"Loading per-channel {obj_name(t)} not supported")
                trigger_dict = attr.asdict(t)
            else:
                trigger_dict = dict(t or {})

            if line_color in trigger_dict:
                trigger_dict[line_color] = color2hex(trigger_dict[line_color])

            cfg.trigger = trigger_dict

    def triggers(self, row: int) -> Dict[str, Any]:
        trigger = self.channels[row].trigger
        assert isinstance(trigger, dict)
        return trigger

    # columns
    col_data = [
        Column("wav_path", path_strip_quotes, "", "WAV Path"),
        Column("amplification", float, None, "Amplification\n(override)"),
        Column("trigger_width", int, None, "Trigger Width ×"),
        Column("render_width", int, None, "Render Width ×"),
        Column("line_color", str, None, "Line Color"),
        Column("trigger__edge_strength", float, None),
        Column("trigger__responsiveness", float, None),
        Column("trigger__buffer_falloff", float, None),
    ]

    idx_of_key = {}
    for idx, col in enumerate(col_data):
        idx_of_key[col.key] = idx
    del idx, col

    def columnCount(self, parent: QModelIndex = ...) -> int:
        return len(self.col_data)

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ) -> Union[str, QVariant]:
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                col = section
                try:
                    return self.col_data[col].display_name
                except IndexError:
                    return nope
            else:
                return str(section + 1)
        return nope

    # rows
    def rowCount(self, parent: QModelIndex = ...) -> int:
        return len(self.channels)

    # data
    TRIGGER = "trigger__"

    def data(self, index: QModelIndex, role=Qt.DisplayRole) -> qc.QVariant:
        col = index.column()
        row = index.row()

        if (
            role in [Qt.DisplayRole, Qt.EditRole]
            and index.isValid()
            and row < self.rowCount()
        ):
            data = self.col_data[col]
            key = data.key
            if key.startswith(self.TRIGGER):
                key = behead(key, self.TRIGGER)
                value = self.triggers(row).get(key, "")

            else:
                value = getattr(self.channels[row], key)

            if value == data.default:
                return ""
            if key == "wav_path" and role == Qt.DisplayRole:
                if Path(value).parent != Path():
                    return "..." + Path(value).name
            return str(value)

        return nope

    def setData(self, index: QModelIndex, value: str, role=Qt.EditRole) -> bool:
        col = index.column()
        row = index.row()

        if index.isValid() and role == Qt.EditRole:
            # type(value) == str

            data = self.col_data[col]
            key = data.key
            if value and not value.isspace():
                try:
                    value = data.cls(value)
                except ValueError:
                    return False
            else:
                value = data.default

            if key.startswith(self.TRIGGER):
                key = behead(key, self.TRIGGER)
                trigger = self.triggers(row)
                if value == data.default:
                    # Delete key if (key: value) present
                    trigger.pop(key, None)
                else:
                    trigger[key] = value

            else:
                setattr(self.channels[row], key, value)

            self.dataChanged.emit(index, index, [role])
            return True
        return False

    """So if I understood it correctly you want to reorder the columns by moving the 
    headers and then want to know how the view looks like. I believe ( 90% certain ) 
    when you reorder the headers it does not trigger any change in the model! and 
    then if you just start printing the data of the model you will only see the data 
    in the order how it was initially before you swapper/reordered some column with 
    the header. """

    def insertRows(self, row: int, count: int, parent=QModelIndex()) -> bool:
        if not (count >= 1 and 0 <= row <= len(self.channels)):
            return False

        self.beginInsertRows(parent, row, row + count - 1)
        self.channels[row:row] = [ChannelConfig("") for _ in range(count)]
        self.endInsertRows()
        return True

    def removeRows(self, row: int, count: int, parent=QModelIndex()) -> bool:
        nchan = len(self.channels)
        # row <= nchan for consistency.
        if not (count >= 1 and 0 <= row <= nchan and row + count <= nchan):
            return False

        self.beginRemoveRows(parent, row, row + count - 1)
        del self.channels[row : row + count]
        self.endRemoveRows()
        return True

    def moveRows(
        self,
        _sourceParent: QModelIndex,
        src_row: int,
        count: int,
        _destinationParent: QModelIndex,
        dest_row: int,
    ):
        nchan = len(self.channels)
        if not (
            count >= 1
            and 0 <= src_row <= nchan
            and src_row + count <= nchan
            and 0 <= dest_row <= nchan
        ):
            return False

        # If source and destination overlap, beginMoveRows returns False.
        if not self.beginMoveRows(
            _sourceParent, src_row, src_row + count - 1, _destinationParent, dest_row
        ):
            return False

        # We know source and destination do not overlap.
        src = slice(src_row, src_row + count)
        dest = slice(dest_row, dest_row)

        if dest_row > src_row:
            # Move down: Insert dest, then remove src
            self.channels[dest] = self.channels[src]
            del self.channels[src]
        else:
            # Move up: Remove src, then insert dest.
            rows = self.channels[src]
            del self.channels[src]
            self.channels[dest] = rows
        self.endMoveRows()
        return True

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.ItemIsEnabled
        return (
            qc.QAbstractItemModel.flags(self, index)
            | Qt.ItemIsEditable
            | Qt.ItemNeverHasChildren
        )


nope = qc.QVariant()


class DownloadFFmpegActivity:
    title = "Missing FFmpeg"

    ffmpeg_url = paths.get_ffmpeg_url()
    can_download = bool(ffmpeg_url)

    path_uri = qc.QUrl.fromLocalFile(paths.PATH_dir).toString()

    required = (
        f"FFmpeg+FFplay must be in PATH or "
        f'<a href="{path_uri}">corrscope folder</a> in order to use corrscope.<br>'
    )

    ffmpeg_template = required + (
        f'Download ffmpeg from <a href="{ffmpeg_url}">{ffmpeg_url}</a>.'
    )
    fail_template = required + "Cannot download FFmpeg for your platform."

    def __init__(self, window: qw.QWidget):
        """Prompt the user to download and install ffmpeg."""
        Msg = qw.QMessageBox

        if not self.can_download:
            Msg.information(window, self.title, self.fail_template, Msg.Ok)
            return

        Msg.information(window, self.title, self.ffmpeg_template, Msg.Ok)
