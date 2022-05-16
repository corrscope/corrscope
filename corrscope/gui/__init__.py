import functools
import signal
import sys
import traceback
from contextlib import contextmanager
from enum import Enum
from threading import Thread
from pathlib import Path
from types import MethodType
from typing import Optional, List, Any, Tuple, Callable, Union, Dict, Sequence, NewType

import appnope
import qtpy.QtCore as qc
import qtpy.QtWidgets as qw
import attr
from qtpy.QtCore import QModelIndex, Qt, QVariant
from qtpy.QtGui import QFont, QCloseEvent, QDesktopServices

import corrscope
import corrscope.settings.global_prefs as gp
from corrscope import cli
from corrscope.channel import ChannelConfig, DefaultLabel
from corrscope.config import CorrError, copy_config, yaml
from corrscope.corrscope import CorrScope, Config, Arguments, template_config
from corrscope.gui.history_file_dlg import (
    get_open_file_name,
    get_open_file_list,
    get_save_file_path,
)
from corrscope.gui.model_bind import (
    PresentationModel,
    map_gui,
    behead,
    rgetattr,
    rsetattr,
    Symbol,
    SymbolText,
    BoundComboBox,
)
from corrscope.gui.util import color2hex, Locked, find_ranges, TracebackDialog
from corrscope.gui.version_common import QT6
from corrscope.gui.view_mainwindow import MainWindow as Ui_MainWindow
from corrscope.gui.widgets import ChannelTableView, ShortcutButton
from corrscope.layout import Orientation, StereoOrientation
from corrscope.outputs import IOutputConfig, FFplayOutputConfig
from corrscope.renderer import LabelPosition
from corrscope.settings import paths
from corrscope.triggers import (
    CorrelationTriggerConfig,
    MainTriggerConfig,
    SpectrumConfig,
    ZeroCrossingTriggerConfig,
)
from corrscope.util import obj_name, iround, coalesce
from corrscope.wave import Flatten

FILTER_WAV_FILES = ["WAV files (*.wav)"]
FILTER_IMAGES = ["Images files (*.png *.jpg *.jpeg *.gif)", "All files (*)"]

APP_NAME = f"{corrscope.app_name} {corrscope.__version__}"
APP_DIR = Path(__file__).parent

PATH_uri = qc.QUrl.fromLocalFile(paths.PATH_dir)


def res(file: str) -> str:
    return str(APP_DIR / file)


@contextmanager
def exception_as_dialog(window: qw.QWidget):
    def excepthook(exc_type, exc_val, exc_tb):
        TracebackDialog(window).showMessage(format_stack_trace(exc_val))

    orig = sys.excepthook
    try:
        sys.excepthook = excepthook
        yield
    finally:
        sys.excepthook = orig


def gui_main(cfg_or_path: Union[Config, Path]):
    # Allow Ctrl-C to exit
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # qw.QApplication.setStyle('fusion')
    QApp = qw.QApplication
    if not QT6:
        QApp.setAttribute(qc.Qt.AA_EnableHighDpiScaling)

    app = qw.QApplication(sys.argv)

    # On Windows, Qt 5's default system font (MS Shell Dlg 2) is outdated.
    # Interestingly, the QMenu font is correct and comes from lfMenuFont.
    # So use it for the entire application.
    # Qt on Windows will finally switch default font to lfMessageFont=Segoe UI
    # (Vista, 2006)... in 2020 (Qt 6.0).
    if qc.QSysInfo.kernelType() == "winnt":
        QApp.setFont(QApp.font("QMenu"))

    window = MainWindow(cfg_or_path)

    # Any exceptions raised within MainWindow() will be caught within exec_.
    # exception_as_dialog() turns it into a Qt dialog.
    with exception_as_dialog(window):
        ret = app.exec_()
        # Any exceptions raised after exec_ terminates will call
        # exception_as_dialog().__exit__ before being caught.
        # This produces a Python traceback.

    # On Linux, if signal.signal(signal.SIGINT, signal.SIG_DFL) and Ctrl+C pressed,
    # corrscope closes immediately.
    # ffmpeg receives SIGPIPE and terminates by itself (according to strace).
    corr_thread = window.corr_thread
    if corr_thread is not None:
        corr_thread.job.abort_terminate()
        corr_thread.join()

    sys.exit(ret)


SafeProperty = NewType("SafeProperty", property)


def safe_property(unsafe_getter: Callable, *args, **kwargs) -> SafeProperty:
    """Prevents (AttributeError from leaking outside a property,
    which causes hasattr() to return False)."""

    @functools.wraps(unsafe_getter)
    def safe_getter(self):
        try:
            return unsafe_getter(self)
        except AttributeError as e:
            raise RuntimeError(e) from e

    # NewType("", cls)(x) == x.
    return SafeProperty(property(safe_getter, *args, **kwargs))


class MainWindow(qw.QMainWindow, Ui_MainWindow):
    """
    Main window.

    Control flow:
    __init__: either
    - load_cfg
    - load_cfg_from_path

    Opening a document:
    - load_cfg_from_path

    ## Dialog Directory/Filename Generation

    Save-dialog dir is persistent state, saved across program runs.
    Most recent of:
    - Any open/save dialog (unless separate_render_dir is True).
        - self.pref.file_dir_ref, .set()
    - Load YAML from CLI.
        - load_cfg_from_path(cfg_path) sets `self.pref.file_dir`.
    - Load .wav files from CLI.
        - if isinstance(cfg_or_path, Config):
            - save_dir = self.compute_save_dir(self.cfg)
            - self.pref.file_dir = save_dir (if not empty)

    Render-dialog dir is persistent state, = most recent render-save dialog.
    - self.pref.render_dir, .set()

    Save/render-dialog filename (no dir) is computed on demand, NOT persistent state.
    - (Currently loaded config path, or master audio, or channel 0) + ext.
    - Otherwise empty string.
        - self.get_save_filename() calls cli.get_file_stem().

    CLI YAML filename is the same,
    but defaults to "corrscope.yaml" instead of empty string.
    - cli._get_file_name() calls cli.get_file_stem().

    CLI video filename is explicitly specified by the user.
    """

    def __init__(self, cfg_or_path: Union[Config, Path]):
        super().__init__()

        # Load settings.
        prefs_error = None
        try:
            self.pref = gp.load_prefs()
            if not isinstance(self.pref, gp.GlobalPrefs):
                raise TypeError(f"prefs.yaml contains wrong type {type(self.pref)}")
        except Exception as e:
            prefs_error = e
            self.pref = gp.GlobalPrefs()

        # Load UI.
        self.setupUi(self)  # sets windowTitle

        # Bind UI buttons, etc. Functions block main thread, avoiding race conditions.
        self.master_audio_browse.clicked.connect(self.on_master_audio_browse)
        self.bg_image_browse.clicked.connect(self.on_bg_image_browse)

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

        self.action_open_config_dir.triggered.connect(self.on_open_config_dir)

        self.actionNew.triggered.connect(self.on_action_new)
        self.actionOpen.triggered.connect(self.on_action_open)
        self.actionSave.triggered.connect(self.on_action_save)
        self.actionSaveAs.triggered.connect(self.on_action_save_as)
        self.actionPreview.triggered.connect(self.on_action_preview)
        self.actionRender.triggered.connect(self.on_action_render)

        self.actionWebsite.triggered.connect(self.on_action_website)
        self.actionHelp.triggered.connect(self.on_action_help)

        self.actionExit.triggered.connect(qw.QApplication.closeAllWindows)

        # Initialize CorrScope-thread attribute.
        self.corr_thread: Optional[CorrThread] = None

        # Setup UI.
        self.model = ConfigModel(template_config())
        self.model.edited.connect(self.on_model_edited)
        # Calls self.on_gui_edited() whenever GUI widgets change.
        map_gui(self, self.model)

        self.model.update_widget["render_stereo"].append(self.on_render_stereo_changed)

        # Bind config to UI.
        if isinstance(cfg_or_path, Config):
            self.load_cfg(cfg_or_path, None)
            save_dir = self.compute_save_dir(self.cfg)
            if save_dir:
                self.pref.file_dir = save_dir
        elif isinstance(cfg_or_path, Path):
            self.load_cfg_from_path(cfg_or_path)
        else:
            raise TypeError(
                f"argument cfg={cfg_or_path} has invalid type {obj_name(cfg_or_path)}"
            )

        self.show()

        if prefs_error is not None:
            TracebackDialog(self).showMessage(
                "Warning: failed to load global preferences, resetting to default.\n"
                + format_stack_trace(prefs_error)
            )

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

    # Config models
    model: Optional["ConfigModel"] = None

    channel_model: "ChannelModel"
    channel_view: "ChannelTableView"
    channelsGroup: qw.QGroupBox

    def on_render_stereo_changed(self):
        self.layout__stereo_orientation.setEnabled(
            self.model.cfg.render_stereo is Flatten.Stereo
        )

    # Closing active document

    def _cancel_render_if_active(self, title: str) -> bool:
        """
        :return: False if user cancels close-document action.
        """
        if self.corr_thread is None:
            return True

        Msg = qw.QMessageBox

        message = self.tr("Cancel current {} and close project?").format(
            self.preview_or_render
        )
        response = Msg.question(self, title, message, Msg.Yes | Msg.No, Msg.No)

        if response == Msg.Yes:
            # Closing ffplay preview (can't cancel render, the dialog is untouchable)
            # will set self.corr_thread to None while the dialog is active.
            # https://www.vikingsoftware.com/how-to-use-qthread-properly/ # QObject thread affinity
            # But since the dialog is modal,
            # self.corr_thread cannot have been replaced by a different thread.
            if self.corr_thread is not None:
                self.corr_thread.abort_terminate()
            return True

        return False

    def _prompt_if_unsaved(self, title: str) -> bool:
        """
        :return: False if user cancels close-document action.
        """
        if not self.any_unsaved:
            return True

        Msg = qw.QMessageBox

        message = f"Save changes to {self.title_cache}?"
        should_close = Msg.question(
            self, title, message, Msg.Save | Msg.Discard | Msg.Cancel
        )

        if should_close == Msg.Cancel:
            return False
        elif should_close == Msg.Discard:
            return True
        else:
            return self.on_action_save()

    def should_close_document(self, title: str) -> bool:
        """
        Called when user is closing document
        (when opening a new document or closing the app).

        :return: False if user cancels close-document action.
        """
        if not self._prompt_if_unsaved(title):
            return False
        if not self._cancel_render_if_active(title):
            # Saying Yes quits render immediately, so place this dialog last.
            return False
        return True

    def closeEvent(self, event: QCloseEvent) -> None:
        """Called on closing window."""
        if self.should_close_document(self.tr("Quit")):
            gp.dump_prefs(self.pref)
            event.accept()
        else:
            event.ignore()

    def on_action_new(self):
        if not self.should_close_document(self.tr("New Project")):
            return
        cfg = template_config()
        self.load_cfg(cfg, None)

    def on_action_open(self):
        if not self.should_close_document(self.tr("Open Project")):
            return
        name = get_open_file_name(
            self, "Open config", self.pref.file_dir_ref, ["YAML files (*.yaml)"]
        )
        if name:
            cfg_path = Path(name)
            self.load_cfg_from_path(cfg_path)

    def load_cfg_from_path(self, cfg_path: Path):
        # Bind GUI to dummy config, in case loading cfg_path raises Exception.
        if self.model is None:
            self.load_cfg(template_config(), None)

        assert cfg_path.is_file()
        self.pref.file_dir = str(cfg_path.parent.resolve())

        # Raises YAML structural exceptions
        cfg = yaml.load(cfg_path)

        try:
            # Raises color getter exceptions
            self.load_cfg(cfg, cfg_path)
        except Exception as e:
            # FIXME if error halfway, clear "file path" and load empty model.
            TracebackDialog(self).showMessage(format_stack_trace(e))
            return

    def load_cfg(self, cfg: Config, cfg_path: Optional[Path]) -> None:
        self._cfg_path = cfg_path
        self._any_unsaved = False
        self.load_title()
        self.left_tabs.setCurrentIndex(0)

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
    action_open_config_dir: qw.QAction

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
            self, "Open master audio file", self.pref.file_dir_ref, FILTER_WAV_FILES
        )
        if name:
            master_audio = "master_audio"
            self.model[master_audio] = name
            self.model.update_all_bound(master_audio)

    def on_bg_image_browse(self):
        name = get_open_file_name(
            self, "Open background image file", self.pref.file_dir_ref, FILTER_IMAGES
        )
        if name:
            bg_image = "render__bg_image"
            self.model[bg_image] = name
            self.model.update_all_bound(bg_image)

    def on_separate_render_dir_toggled(self, checked: bool):
        self.pref.separate_render_dir = checked
        if checked:
            self.pref.render_dir = self.pref.file_dir
        else:
            self.pref.render_dir = ""

    def on_open_config_dir(self):
        appdata_uri = qc.QUrl.fromLocalFile(str(paths.appdata_dir))
        QDesktopServices.openUrl(appdata_uri)

    def on_channel_add(self):
        wavs = get_open_file_list(
            self, "Add audio channels", self.pref.file_dir_ref, FILTER_WAV_FILES
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

        # Name and extension (no folder).
        cfg_filename = self.get_save_filename(cli.YAML_NAME)

        # Folder is obtained from self.pref.file_dir_ref.
        filters = ["YAML files (*.yaml)", "All files (*)"]
        path = get_save_file_path(
            self,
            "Save As",
            self.pref.file_dir_ref,
            cfg_filename,
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
        """Launch CorrScope and ffplay."""
        if self.corr_thread is not None:
            error_msg = self.tr("Cannot preview, another {} is active").format(
                self.preview_or_render
            )
            qw.QMessageBox.critical(self, "Error", error_msg)
            return

        outputs = [FFplayOutputConfig()]
        self.play_thread(outputs, PreviewOrRender.preview, dlg=None)

    def on_action_render(self):
        """Get file name. Then show a progress dialog while rendering to file."""
        if self.corr_thread is not None:
            error_msg = self.tr("Cannot render to file, another {} is active").format(
                self.preview_or_render
            )
            qw.QMessageBox.critical(self, "Error", error_msg)
            return

        # Name and extension (no folder).
        video_filename = self.get_save_filename(cli.VIDEO_NAME)
        filters = [
            "MP4 files (*.mp4)",
            "Matroska files (*.mkv)",
            "WebM files (*.webm)",
            "All files (*)",
        ]

        # Points to either `file_dir` or `render_dir`.
        # Folder is obtained from `dir_ref`.
        dir_ref = self.pref.render_dir_ref

        path = get_save_file_path(
            self, "Render to Video", dir_ref, video_filename, filters, cli.VIDEO_NAME
        )
        if path:
            name = str(path)
            dlg = CorrProgressDialog(self, "Rendering video")

            # FFmpegOutputConfig contains only hashable/immutable strs,
            # so get_ffmpeg_cfg() can be shared across threads without copying.
            # Optionally copy_config() first.

            outputs = [self.cfg.get_ffmpeg_cfg(name)]
            self.play_thread(outputs, PreviewOrRender.render, dlg)

    def play_thread(
        self,
        outputs: List[IOutputConfig],
        mode: "PreviewOrRender",
        dlg: Optional["CorrProgressDialog"],
    ):
        assert self.model

        arg = self._get_args(outputs)
        cfg = copy_config(self.model.cfg)
        t = self.corr_thread = CorrThread(cfg, arg, mode)

        if dlg:
            # t.abort -> Locked.set() is thread-safe (hopefully).
            # It can be called from main thread (not just within CorrThread).
            dlg.canceled.connect(t.job.abort, Qt.DirectConnection)
            t.job.arg = attr.evolve(
                arg,
                on_begin=run_on_ui_thread(dlg.on_begin, (float, float)),
                progress=run_on_ui_thread(dlg.setValue, (int,)),
                on_end=run_on_ui_thread(dlg.reset, ()),  # TODO dlg.close
            )

        t.job.finished.connect(self.on_play_thread_finished)
        t.job.error.connect(self.on_play_thread_error)
        t.job.ffmpeg_missing.connect(self.on_play_thread_ffmpeg_missing)
        t.start()

    @safe_property
    def preview_or_render(self) -> str:
        if self.corr_thread is not None:
            return self.tr(self.corr_thread.job.mode.value)
        return "neither preview nor render"

    def _get_args(self, outputs: List[IOutputConfig]):
        def raise_exception():
            raise RuntimeError(
                "Arguments.is_aborted should be overwritten by CorrThread"
            )

        arg = Arguments(
            cfg_dir=self.cfg_dir, outputs=outputs, is_aborted=raise_exception
        )
        return arg

    def on_play_thread_finished(self):
        self.corr_thread = None

    def on_play_thread_error(self, stack_trace: str):
        TracebackDialog(self).showMessage(stack_trace)

    def on_play_thread_ffmpeg_missing(self):
        DownloadFFmpegActivity(self)

    # File paths
    @safe_property
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

    @safe_property
    def title(self) -> str:
        if self._cfg_path:
            return self._cfg_path.name
        return self.UNTITLED

    def get_save_filename(self, suffix: str) -> str:
        """
        If file name can be guessed, return "filename.suffix" (no dir).
        Otherwise return "".

        Used for saving file or video.
        """
        stem = cli.get_file_stem(self._cfg_path, self.cfg, default="")
        if stem:
            return stem + suffix
        else:
            return ""

    @staticmethod
    def compute_save_dir(cfg: Config) -> Optional[str]:
        """Computes a "save directory" when constructing a config from CLI wav files."""
        if cfg.master_audio:
            file_path = cfg.master_audio
        elif len(cfg.channels) > 0:
            file_path = cfg.channels[0].wav_path
        else:
            return None

        # If file_path is "file.wav", we want to return "." .
        # os.path.dirname("file.wav") == ""
        # Path("file.wav").parent..str == "."
        dir = Path(file_path).parent
        return str(dir)

    @safe_property
    def cfg(self):
        return self.model.cfg

    # Misc.
    @qc.Slot()
    def on_action_website(self):
        website_url = r"https://github.com/corrscope/corrscope/"
        QDesktopServices.openUrl(qc.QUrl(website_url))

    @qc.Slot()
    def on_action_help(self):
        help_url = r"https://corrscope.github.io/corrscope/"
        QDesktopServices.openUrl(qc.QUrl(help_url))


def _format_exc_value(e: BaseException, limit=None, chain=True):
    """Like traceback.format_exc() but takes an exception object."""
    list = traceback.format_exception(
        type(e), e, e.__traceback__, limit=limit, chain=chain
    )
    str = "".join(list)
    return str


def format_stack_trace(e: BaseException):
    if isinstance(e, CorrError):
        stack_trace = _format_exc_value(e, limit=0)
    else:
        stack_trace = _format_exc_value(e)
    return stack_trace


class PreviewOrRender(Enum):
    # PreviewOrRender.value is translated at time of use, not time of definition.
    preview = qc.QT_TRANSLATE_NOOP("MainWindow", "preview")
    render = qc.QT_TRANSLATE_NOOP("MainWindow", "render")


class CorrJob(qc.QObject):
    is_aborted: Locked[bool]

    @qc.Slot()
    def abort(self):
        self.is_aborted.set(True)

    def abort_terminate(self):
        """Sends abort signal to main loop, and terminates all outputs."""
        self.abort()
        if self.corr is not None:
            for output in self.corr.outputs:
                output.terminate(from_same_thread=False)

    finished = qc.Signal()
    error = qc.Signal(str)
    ffmpeg_missing = qc.Signal()

    def __init__(self, cfg: Config, arg: Arguments, mode: PreviewOrRender):
        qc.QObject.__init__(self)
        self.is_aborted = Locked(False)

        self.cfg = cfg
        self.arg = arg
        self.arg.is_aborted = self.is_aborted.get
        self.mode = mode
        self.corr = None  # type: Optional[CorrScope]

    def run(self) -> None:
        """Called in separate thread."""
        cfg = self.cfg
        arg = self.arg
        with appnope.nope_scope(reason="corrscope preview/render active"):
            try:
                self.corr = CorrScope(cfg, arg)
                self.corr.play()

            except paths.MissingFFmpegError:
                arg.on_end()
                self.ffmpeg_missing.emit()

            except Exception as e:
                arg.on_end()
                stack_trace = format_stack_trace(e)
                self.error.emit(stack_trace)

            else:
                arg.on_end()


class CorrThread(Thread):
    job: CorrJob

    def __init__(self, cfg: Config, arg: Arguments, mode: PreviewOrRender):
        Thread.__init__(self)
        self.job = CorrJob(cfg, arg, mode)

    def run(self):
        """Callback invoked on new thread."""
        try:
            self.job.run()
        finally:
            self.job.finished.emit()


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

    @qc.Slot(float, float)
    def on_begin(self, begin_time, end_time):
        self.setRange(iround(begin_time), iround(end_time))
        # self.setValue is called by CorrScope, on the first frame.


def run_on_ui_thread(
    bound_slot: MethodType, types: Tuple[type, ...]
) -> Callable[..., None]:
    """Runs an object's slot on the object's own thread.
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


# Begin ConfigModel properties


def nrow_ncol_property(altered: str, unaltered: str) -> SafeProperty:
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
            self.update_all_bound("layout__" + unaltered)
        elif val == 0:
            setattr(self.cfg.layout, altered, None)
        else:
            raise CorrError(f"invalid input: {altered} < 0, should never happen")

    return safe_property(get, set)


# Unused
def default_property(path: str, default: Any) -> SafeProperty:
    def getter(self: "ConfigModel"):
        val = rgetattr(self.cfg, path)
        if val is None:
            return default
        else:
            return val

    def setter(self: "ConfigModel", val):
        rsetattr(self.cfg, path, val)

    return safe_property(getter, setter)


def path_strip_quotes(path: str) -> str:
    if len(path) and path[0] == path[-1] == '"':
        return path[1:-1]
    return path


def path_fix_property(path: str) -> SafeProperty:
    """Removes quotes from paths, when setting from GUI."""

    def getter(self: "ConfigModel") -> str:
        return rgetattr(self.cfg, path)

    def setter(self: "ConfigModel", val: str):
        val = path_strip_quotes(val)
        rsetattr(self.cfg, path, val)

    return safe_property(getter, setter)


flatten_no_stereo = {
    Flatten.SumAvg: "Average: (L+R)/2",
    Flatten.DiffAvg: "DiffAvg: (L-R)/2",
}
flatten_modes = {**flatten_no_stereo, Flatten.Stereo: "Stereo"}
assert set(flatten_modes.keys()) == set(Flatten.modes)  # type: ignore


class ConfigModel(PresentationModel):
    cfg: Config
    combo_symbol_text: Dict[str, Sequence[SymbolText]] = {}

    master_audio = path_fix_property("master_audio")

    # Stereo flattening
    combo_symbol_text["trigger_stereo"] = list(flatten_no_stereo.items()) + [
        (BoundComboBox.Custom, "Custom")
    ]
    combo_symbol_text["render_stereo"] = list(flatten_modes.items()) + [
        (BoundComboBox.Custom, "Custom")
    ]

    # Trigger
    @safe_property
    def trigger__pitch_tracking(self) -> bool:
        scfg = self.cfg.trigger.pitch_tracking
        gui = scfg is not None
        return gui

    @trigger__pitch_tracking.setter
    def trigger__pitch_tracking(self, gui: bool):
        scfg = SpectrumConfig() if gui else None
        self.cfg.trigger.pitch_tracking = scfg

    combo_symbol_text["trigger__edge_direction"] = [
        (1, "Rising (+1)"),
        (-1, "Falling (-1)"),
    ]

    combo_symbol_text["trigger__post_trigger"] = [
        (type(None), "Disabled"),
        (ZeroCrossingTriggerConfig, "Zero Crossing"),
    ]

    # Render
    @safe_property
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

    combo_symbol_text["default_label"] = [
        (DefaultLabel.NoLabel, MainWindow.tr("None", "Default Label")),
        (DefaultLabel.FileName, MainWindow.tr("File Name", "Default Label")),
        (DefaultLabel.Number, MainWindow.tr("Number", "Default Label")),
    ]

    combo_symbol_text["render.label_position"] = [
        (LabelPosition.LeftTop, "Top Left"),
        (LabelPosition.LeftBottom, "Bottom Left"),
        (LabelPosition.RightTop, "Top Right"),
        (LabelPosition.RightBottom, "Bottom Right"),
    ]

    @safe_property
    def render__label_qfont(self) -> QFont:
        qfont = QFont()
        qfont.setStyleHint(QFont.SansSerif)  # no-op on X11

        font = self.cfg.render.label_font
        if font.toString:
            qfont.fromString(font.toString)
            return qfont

        # Passing None or "" to QFont(family) results in qfont.family() = "", and
        # wrong font being selected (Abyssinica SIL, which appears early in the list).
        family = coalesce(font.family, qfont.defaultFamily())
        # Font file selection
        qfont.setFamily(family)
        qfont.setBold(font.bold)
        qfont.setItalic(font.italic)
        # Font size
        qfont.setPointSizeF(font.size)
        return qfont

    @render__label_qfont.setter
    def render__label_qfont(self, qfont: QFont):
        self.cfg.render.label_font = attr.evolve(
            self.cfg.render.label_font,
            # Font file selection
            family=qfont.family(),
            bold=qfont.bold(),
            italic=qfont.italic(),
            # Font size
            size=qfont.pointSizeF(),
            # QFont implementation details
            toString=qfont.toString(),
        )

    # Layout
    layout__nrows = nrow_ncol_property("nrows", unaltered="ncols")
    layout__ncols = nrow_ncol_property("ncols", unaltered="nrows")

    _orientations = [["h", "Horizontal"], ["v", "Vertical"]]
    _stereo_orientations = _orientations + [["overlay", "Overlay"]]

    combo_symbol_text["layout__orientation"] = [
        (Orientation(key), name) for key, name in _orientations
    ]
    combo_symbol_text["layout__stereo_orientation"] = [
        (StereoOrientation(key), name) for key, name in _stereo_orientations
    ]


# End ConfigModel


@attr.dataclass
class Column:
    key: str

    # fn(str) -> T. If ValueError is thrown, replaced by `default`.
    cls: Union[type, Callable[[str], Any]]

    # `default` is written into config,
    # when users type "blank or whitespace" into table cell.
    default: Any

    def _display_name(self) -> str:
        return self.key.replace("__", "\n").replace("_", " ").title()

    display_name: str = attr.Factory(_display_name, takes_self=True)
    always_show: bool = False


def plus_minus_one(value: str) -> int:
    if int(value) >= 0:  # Raises ValueError
        return 1
    else:
        return -1


nope = qc.QVariant()


def parse_bool_maybe(s: str) -> Optional[bool]:
    """Does not throw. But could legally throw ValueError."""

    if not s:
        return None

    # len(s) >= 1
    try:
        return bool(int(s))
    except ValueError:
        pass

    s = s.lower()
    if s[0] in ["t", "y"]:
        return True
    if s[0] in ["f", "n"]:
        return False
    return None


class ChannelModel(qc.QAbstractTableModel):
    """Design based off
    https://doc.qt.io/qt-5/model-view-programming.html#a-read-only-example-model and
    https://doc.qt.io/qt-5/model-view-programming.html#model-subclassing-reference
    """

    def __init__(self, channels: List[ChannelConfig]):
        """Mutates `channels` and `line_color` for convenience."""
        super().__init__()
        self.channels = channels

        line_color = "line_color"

        for cfg in self.channels:
            t = cfg.trigger
            if isinstance(t, MainTriggerConfig):
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
        Column("label", str, "", "Label"),
        Column("amplification", float, None, "Amplification\n(override)"),
        Column("line_color", str, None, "Line Color"),
        Column("color_by_pitch", parse_bool_maybe, None, "Color Lines\nBy Pitch"),
        Column("render_stereo", str, None, "Render Stereo\nDownmix"),
        Column("trigger_width", int, 1, "Trigger Width ×", always_show=True),
        Column("render_width", int, 1, "Render Width ×", always_show=True),
        Column("trigger__mean_responsiveness", float, None, "DC Removal\nRate"),
        Column("trigger__sign_strength", float, None, "Sign\nAmplification"),
        Column("trigger__edge_direction", plus_minus_one, None),
        Column("trigger__edge_strength", float, None),
        Column("trigger__slope_width", float, None),
        Column("trigger__buffer_strength", float, None),
        Column("trigger__responsiveness", float, None, "Buffer\nResponsiveness"),
        Column("trigger__reset_below", float, None, "Reset Below\nMatch"),
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

    def data(self, index: QModelIndex, role=Qt.DisplayRole) -> Any:
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

            if not data.always_show and value == data.default:
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


class DownloadFFmpegActivity:
    title = "Missing FFmpeg"

    ffmpeg_url = paths.get_ffmpeg_url()
    can_download = bool(ffmpeg_url)

    required = (
        f"FFmpeg+FFplay must be in PATH or "
        f'<a href="{PATH_uri.toString()}">corrscope PATH</a> in order to use corrscope.<br>'
    )

    ffmpeg_template = required + (
        f'Download ffmpeg from <a href="{ffmpeg_url}">this link</a>, '
        f"open in 7-Zip and navigate to the ffmpeg-.../bin folder, "
        f"and copy all .exe files to the folder above."
    )
    fail_template = required + "Cannot download FFmpeg for your platform."

    def __init__(self, window: qw.QWidget):
        """Prompt the user to download and install ffmpeg."""
        Msg = qw.QMessageBox

        if not self.can_download:
            Msg.information(window, self.title, self.fail_template, Msg.Ok)
            return

        Msg.information(window, self.title, self.ffmpeg_template, Msg.Ok)
