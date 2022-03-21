# -*- coding: utf-8 -*-
from contextlib import contextmanager

from qtpy.QtCore import *
from qtpy.QtWidgets import *

from corrscope.gui.view_stack import (
    LayoutStack,
    set_layout,
    central_widget,
    append_widget,
    add_row,
    add_tab,
    set_attr_objectName,
    append_stretch,
    Both,
    set_menu_bar,
    append_menu,
    add_toolbar,
    create_element,
    fill_scroll_stretch,
)

NBSP = "\xa0"


class HLine(QFrame):
    def __init__(self, parent):
        super(HLine, self).__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class VLine(QFrame):
    def __init__(self, parent):
        super(VLine, self).__init__(parent)
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)


def fixed_size_policy():
    return QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


# noinspection PyAttributeOutsideInit
class MainWindow(QWidget):
    @staticmethod
    def tr(*args, **kwargs) -> str:
        """Only at runtime, not at pylupdate5 time."""
        # noinspection PyCallByClass,PyTypeChecker
        return QCoreApplication.translate("MainWindow", *args, **kwargs)

    left_tabs: "TabWidget"

    def setupUi(self, MainWindow: QMainWindow):
        # Multiplying by DPI scale is necessary on both Windows and Linux,
        # since MainWindow.resize() operates in physical pixels.
        scale = MainWindow.logicalDpiX() / 96.0

        width = 1280
        height = 0
        MainWindow.resize(int(width * scale), int(height * scale))

        s = LayoutStack(MainWindow)

        # Window contents
        with central_widget(s, QWidget) as self.centralWidget:
            horizontalLayout = set_layout(s, QHBoxLayout)

            # Left-hand config tabs
            with append_widget(s, TabWidget) as self.left_tabs:
                self.add_general_tab(s)
                self.add_appear_tab(s)
                self.add_trigger_tab(s)

            # Right-hand channel list
            with append_widget(s, QVBoxLayout) as self.audioColumn:

                # Top bar (master audio, trigger)
                self.add_top_bar(s)

                # Channel list (group box)
                self.channelsGroup = self.add_channels_list(s)

            # Right-hand channel list expands to fill space.
            horizontalLayout.setStretch(1, 1)

        self.add_actions(s, MainWindow)

        # Creates references to labels
        set_attr_objectName(self, s)

        # Initializes labels by reference
        self.retranslateUi(MainWindow)

    def add_general_tab(self, s: LayoutStack) -> QWidget:
        tr = self.tr
        with self.add_tab_stretch(s, tr("&General"), layout=QVBoxLayout) as tab:

            # Global group
            with append_widget(s, QGroupBox) as self.optionGlobal:
                set_layout(s, QFormLayout)

                with add_row(s, "", BoundSpinBox) as self.fps:
                    self.fps.setMinimum(1)
                    self.fps.setMaximum(999)
                    self.fps.setSingleStep(10)

                with add_row(s, "", BoundSpinBox, maximum=200) as self.trigger_ms:
                    self.trigger_ms.setMinimum(5)
                    self.trigger_ms.setSingleStep(5)

                with add_row(s, "", BoundSpinBox, maximum=200) as self.render_ms:
                    self.render_ms.setMinimum(5)
                    self.render_ms.setSingleStep(5)

                with add_row(s, "", BoundDoubleSpinBox) as self.amplification:
                    self.amplification.setSingleStep(0.1)

                with add_row(s, "", BoundDoubleSpinBox) as self.begin_time:
                    self.begin_time.setMaximum(9999.0)

            with append_widget(
                s, QGroupBox, title=tr("Performance (Preview Only)"), layout=QFormLayout
            ):
                with add_row(
                    s, tr("Render Subsampling"), BoundSpinBox
                ) as self.render_subsampling:
                    self.render_subsampling.setMinimum(1)

                with add_row(
                    s, tr("Render FPS Divisor"), BoundSpinBox
                ) as self.render_subfps:
                    self.render_subfps.setMinimum(1)

                with add_row(
                    s, tr("Resolution Divisor"), BoundDoubleSpinBox
                ) as self.render__res_divisor:
                    self.render__res_divisor.setMinimum(1.0)
                    self.render__res_divisor.setSingleStep(0.5)

        return tab

    def add_appear_tab(self, s: LayoutStack) -> QWidget:
        tr = self.tr

        # Qt Designer produces path "QTabWidget/QWidget/QScrollView/QWidget/items".
        # My current code produces path "QTabWidget/QScrollView/QWidget/items".
        # This is missing a gap between the tab and scroll-area, but saves space.
        with add_tab(
            s, VerticalScrollArea, tr("&Appearance")
        ) as tab, fill_scroll_stretch(s, layout=QVBoxLayout):

            with append_widget(
                s, QGroupBox, title=tr("Appearance"), layout=QFormLayout
            ):
                with add_row(s, "", BoundLineEdit) as self.render_resolution:
                    pass

                with add_row(s, "", BoundColorWidget) as self.render__bg_color:
                    pass

                with add_row(s, tr("Background image"), QHBoxLayout):
                    with append_widget(s, BoundLineEdit) as self.render__bg_image:
                        pass
                    with append_widget(s, QPushButton) as self.bg_image_browse:
                        self.bg_image_browse.setText(tr("&Browse..."))

                with add_row(s, "", BoundColorWidget) as self.render__init_line_color:
                    pass

                with add_row(
                    s, BoundCheckBox, Both
                ) as self.render__global_color_by_pitch:
                    self.render__global_color_by_pitch.setText(
                        tr("Color Lines By Pitch")
                    )

                with add_row(s, "", BoundDoubleSpinBox) as self.render__line_width:
                    self.render__line_width.setMinimum(0.5)
                    self.render__line_width.setSingleStep(0.5)

                with add_row(
                    s, tr("Outline Color"), BoundColorWidget
                ) as self.render__global_line_outline_color:
                    pass

                with add_row(
                    s, tr("Outline Width"), BoundDoubleSpinBox
                ) as self.render__line_outline_width:
                    self.render__line_outline_width.setSingleStep(0.5)

                with add_row(s, "", OptionalColorWidget) as self.render__grid_color:
                    pass

                with add_row(s, "", BoundColorWidget) as self.render__midline_color:
                    pass

                with add_row(s, BoundCheckBox, BoundCheckBox) as (
                    self.render__v_midline,
                    self.render__h_midline,
                ):
                    pass

                with add_row(
                    s,
                    tr("Grid Line Width"),
                    BoundDoubleSpinBox,
                    name="render.grid_line_width",
                    minimum=0.5,
                    singleStep=0.5,
                ):
                    pass

            with append_widget(s, QGroupBox, title=tr("Labels"), layout=QFormLayout):
                with add_row(
                    s, tr("Default Label"), BoundComboBox, name="default_label"
                ):
                    pass
                with add_row(
                    s, tr("Font"), BoundFontButton, name="render__label_qfont"
                ):
                    pass
                with add_row(
                    s,
                    tr("Font Color"),
                    OptionalColorWidget,
                    name="render.label_color_override",
                ):
                    pass

                with add_row(
                    s, tr("Label Position"), BoundComboBox, name="render.label_position"
                ):
                    pass
                with add_row(
                    s,
                    tr("Label Padding"),
                    BoundDoubleSpinBox,
                    name="render.label_padding_ratio",
                    maximum=10,
                    singleStep=0.25,
                ):
                    pass

            with append_widget(s, QGroupBox, title=tr("Layout"), layout=QFormLayout):
                with add_row(s, "", BoundComboBox) as self.layout__orientation:
                    pass

                with add_row(s, tr("Columns"), QHBoxLayout) as self.layoutDims:
                    with append_widget(s, BoundSpinBox) as self.layout__ncols:
                        self.layout__ncols.setSpecialValueText(NBSP)

                    with append_widget(s, QLabel) as self.layout__nrowsL:
                        pass

                    with append_widget(s, BoundSpinBox) as self.layout__nrows:
                        self.layout__nrows.setSpecialValueText(NBSP)

            with append_widget(s, QGroupBox, title=tr("Stereo"), layout=QFormLayout):
                with add_row(
                    s, tr("Downmix Mode"), BoundComboBox, name="render_stereo"
                ):
                    pass

                with add_row(
                    s, tr("Downmix Vector"), BoundLineEdit, name="render_stereo"
                ):
                    pass

                with add_row(
                    s,
                    tr("Stereo Orientation"),
                    BoundComboBox,
                    name="layout__stereo_orientation",
                ) as self.layout__stereo_orientation:
                    pass

                with add_row(
                    s,
                    tr("Grid Opacity"),
                    BoundDoubleSpinBox,
                    name="render__stereo_grid_opacity",
                    maximum=1.0,
                    singleStep=0.25,
                ):
                    pass

        return tab

    def add_trigger_tab(self, s: LayoutStack) -> QWidget:
        tr = self.tr

        with self.add_tab_stretch(s, tr("&Trigger"), layout=QVBoxLayout) as tab:
            with append_widget(
                s, QGroupBox, title=tr("Input Data Preprocessing"), layout=QFormLayout
            ):
                with add_row(
                    s,
                    tr("DC Removal Rate"),
                    BoundDoubleSpinBox,
                    name="trigger__mean_responsiveness",
                    minimum=0,
                    maximum=1,
                    singleStep=0.25,
                ):
                    pass

                with add_row(
                    s,
                    tr("Sign Amplification\n(for triangle waves)"),
                    BoundDoubleSpinBox,
                    name="trigger__sign_strength",
                    minimum=0,
                    singleStep=0.25,
                ):
                    pass

            with append_widget(
                s, QGroupBox, title=tr("Edge Triggering"), layout=QFormLayout
            ):
                with add_row(
                    s,
                    tr("Trigger Direction"),
                    BoundComboBox,
                    name="trigger__edge_direction",
                ):
                    pass
                with add_row(
                    s,
                    tr("Edge Strength"),
                    BoundDoubleSpinBox,
                    name="trigger__edge_strength",
                ):
                    s.widget.setMinimum(0.0)
                with add_row(
                    s,
                    tr("Slope Width"),
                    BoundDoubleSpinBox,
                    name="trigger__slope_width",
                ):
                    s.widget.setMinimum(0)
                    s.widget.setMaximum(2)
                    s.widget.setSingleStep(0.02)

            with append_widget(
                s, QGroupBox, title=tr("Wave History Alignment"), layout=QFormLayout
            ):
                with add_row(
                    s,
                    tr("Buffer Strength"),
                    BoundDoubleSpinBox,
                    name="trigger__buffer_strength",
                ):
                    pass
                with add_row(
                    s,
                    tr("Buffer Responsiveness"),
                    BoundDoubleSpinBox,
                    name="trigger__responsiveness",
                    maximum=1.0,
                    singleStep=0.1,
                ):
                    pass
                with add_row(
                    s,
                    tr("Reset Below Match"),
                    BoundDoubleSpinBox,
                    name="trigger__reset_below",
                    maximum=1.0,
                    singleStep=0.1,
                ):
                    pass
                with add_row(s, BoundCheckBox, Both) as (self.trigger__pitch_tracking):
                    assert isinstance(self.trigger__pitch_tracking, QWidget)

            with append_widget(
                s, QGroupBox, title=tr("Post Triggering"), layout=QFormLayout
            ):
                with add_row(
                    s, tr("Post Trigger"), TypeComboBox
                ) as self.trigger__post_trigger:
                    pass

                with add_row(
                    s, tr("Post Trigger Radius"), BoundSpinBox
                ) as self.trigger__post_radius:
                    pass
                    # self.trigger__post_radius: BoundSpinBox
                    # self.trigger__post_radius.setMinimum(0)

            with append_widget(
                s,
                QGroupBox,
                title=tr("Stereo (for SNES invert surround)"),
                layout=QFormLayout,
            ):
                with add_row(
                    s, tr("Downmix Mode"), BoundComboBox, name="trigger_stereo"
                ):
                    pass

                with add_row(
                    s, tr("Downmix Vector"), BoundLineEdit, name="trigger_stereo"
                ):
                    pass

        return tab

    @staticmethod
    @contextmanager
    def add_tab_stretch(s: LayoutStack, label: str = "", **kwargs):
        """Create a tab widget,
        then append a stretch to keep GUI elements packed."""
        with add_tab(s, QWidget, label, **kwargs) as tab:
            yield tab
            append_stretch(s)

    def add_top_bar(self, s):
        tr = self.tr
        with append_widget(s, QHBoxLayout):
            with append_widget(s, QVBoxLayout):

                with append_widget(s, QGroupBox):
                    s.widget.setTitle(tr("FFmpeg Options"))
                    set_layout(s, QFormLayout)

                    # Master audio
                    with add_row(s, tr("Master Audio"), QHBoxLayout):
                        with append_widget(s, BoundLineEdit) as self.master_audio:
                            pass
                        with append_widget(s, QPushButton) as self.master_audio_browse:
                            pass

                    with add_row(
                        s, tr("Video Template"), BoundLineEdit
                    ) as self.ffmpeg_cli__video_template:
                        pass
                    with add_row(
                        s, tr("Audio Template"), BoundLineEdit
                    ) as self.ffmpeg_cli__audio_template:
                        pass

    def add_channels_list(self, s):
        tr = self.tr
        with append_widget(s, QGroupBox) as group:
            s.widget.setTitle(tr("Oscilloscope Channels"))
            set_layout(s, QVBoxLayout)

            # Button toolbar
            with append_widget(s, QHBoxLayout) as self.channelBar:
                append_stretch(s)

                with append_widget(s, ShortcutButton) as self.channelAdd:
                    pass
                with append_widget(s, ShortcutButton) as self.channelDelete:
                    pass
                with append_widget(s, ShortcutButton) as self.channelUp:
                    pass
                with append_widget(s, ShortcutButton) as self.channelDown:
                    pass

            # Spreadsheet grid
            with append_widget(s, ChannelTableView) as self.channel_view:
                pass

        return group

    def add_actions(self, s: LayoutStack, MainWindow):
        tr = self.tr
        # Setup actions
        self.actionOpen = QAction(MainWindow)
        self.actionSave = QAction(MainWindow)
        self.actionNew = QAction(MainWindow)
        self.actionSaveAs = QAction(MainWindow)
        self.actionExit = QAction(MainWindow)
        self.actionPreview = QAction(MainWindow)
        self.actionRender = QAction(MainWindow)

        self.actionHelp = create_element(QAction, MainWindow, text=tr("&Help Tutorial"))
        self.actionWebsite = create_element(
            QAction, MainWindow, text=tr("&Project Homepage")
        )

        self.action_separate_render_dir = create_element(
            QAction, MainWindow, text=tr("&Separate Render Folder"), checkable=True
        )
        self.action_open_config_dir = create_element(
            QAction, MainWindow, text=tr("Open &Config Folder")
        )

        # Setup menu_bar
        assert s.widget is MainWindow
        with set_menu_bar(s) as self.menuBar:
            with append_menu(s) as self.menuFile:
                w = self.menuFile
                w.addAction(self.actionNew)
                w.addAction(self.actionOpen)
                w.addAction(self.actionSave)
                w.addAction(self.actionSaveAs)
                w.addSeparator()
                w.addAction(self.actionPreview)
                w.addAction(self.actionRender)
                w.addSeparator()
                w.addAction(self.actionExit)

            with append_menu(s) as self.menuTools:
                w = self.menuTools
                w.addAction(self.action_separate_render_dir)
                w.addSeparator()
                w.addAction(self.action_open_config_dir)

            with append_menu(s, title=tr("&Help")) as self.menuHelp:
                w = self.menuHelp
                w.addAction(self.actionHelp)
                w.addAction(self.actionWebsite)

        # Setup toolbar
        with add_toolbar(s, Qt.TopToolBarArea) as self.toolBar:
            w = self.toolBar
            w.addAction(self.actionNew)
            w.addAction(self.actionOpen)
            w.addAction(self.actionSave)
            w.addAction(self.actionSaveAs)
            w.addSeparator()
            w.addAction(self.actionPreview)
            w.addAction(self.actionRender)

    # noinspection PyUnresolvedReferences
    def retranslateUi(self, MainWindow):
        tr = self.tr

        self.optionGlobal.setTitle(tr("Global"))
        self.fpsL.setText(tr("FPS"))
        self.trigger_msL.setText(tr("Trigger Width"))
        self.render_msL.setText(tr("Render Width"))
        self.amplificationL.setText(tr("Amplification"))
        self.begin_timeL.setText(tr("Begin Time"))
        self.render_resolutionL.setText(tr("Resolution"))
        self.render__bg_colorL.setText(tr("Background"))
        self.render__init_line_colorL.setText(tr("Line Color"))
        self.render__line_widthL.setText(tr("Line Width"))
        self.render__grid_colorL.setText(tr("Grid Color"))
        self.render__midline_colorL.setText(tr("Midline Color"))
        self.render__v_midline.setText(tr("Vertical"))
        self.render__h_midline.setText(tr("Horizontal Midline"))
        self.layout__orientationL.setText(tr("Orientation"))
        self.layout__nrowsL.setText(tr("Rows"))

        self.master_audio_browse.setText(tr("&Browse..."))
        self.trigger__pitch_tracking.setText(tr("Pitch Tracking"))

        self.channelAdd.setText(tr("&Add..."))
        self.channelDelete.setText(tr("&Delete"))
        self.channelUp.setText(tr("Up"))
        self.channelDown.setText(tr("Down"))
        self.menuFile.setTitle(tr("&File"))
        self.menuTools.setTitle(tr("&Tools"))
        self.actionOpen.setText(tr("&Open"))
        self.actionOpen.setShortcut(tr("Ctrl+O"))
        self.actionSave.setText(tr("&Save"))
        self.actionSave.setShortcut(tr("Ctrl+S"))
        self.actionNew.setText(tr("&New"))
        self.actionNew.setShortcut(tr("Ctrl+N"))
        self.actionSaveAs.setText(tr("Save &As"))
        self.actionSaveAs.setShortcut(tr("Ctrl+Shift+S"))
        self.actionExit.setText(tr("E&xit"))
        self.actionExit.setShortcut(tr("Ctrl+Q"))
        self.actionPreview.setText(tr("&Preview"))
        self.actionPreview.setShortcut(tr("Ctrl+P"))
        self.actionRender.setText(tr("&Render to Video"))
        self.actionRender.setShortcut(tr("Ctrl+R"))


from corrscope.gui.model_bind import (
    BoundLineEdit,
    BoundSpinBox,
    BoundDoubleSpinBox,
    BoundCheckBox,
    BoundComboBox,
    TypeComboBox,
    BoundColorWidget,
    OptionalColorWidget,
    BoundFontButton,
)

# Delete unbound widgets, so they cannot accidentally be used.
del QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox

from corrscope.gui.widgets import (
    ChannelTableView,
    ShortcutButton,
    TabWidget,
    VerticalScrollArea,
)

del QTabWidget
