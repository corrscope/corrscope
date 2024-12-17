import os.path
from pathlib import Path
from typing import *

import qtpy.QtWidgets as qw
import attr

import corrscope.settings.global_prefs as _gp

Name = str
Filter = str


T = TypeVar("T")


def _get_hist_name(
    func: Callable[..., Tuple[T, str]],
    parent: qw.QWidget,
    title: str,
    history_dir: _gp.Ref[_gp.GlobalPrefs],
    default_name: Optional[str],
    filters: List[str],
) -> Optional[T]:
    """
    Get file name.
    Default folder is history folder, and `default_name`.folder is discarded.
    If user accepts, update history.
    """
    # Get recently used dir.
    dir_or_file: str = history_dir.get()
    if default_name:
        dir_or_file = os.path.join(dir_or_file, Path(default_name).name)

    # Compute file extension filter.
    filter: str = ";;".join(filters)

    # Call qw.QFileDialog.getXFileName[s].
    name, sel_filter = func(parent, title, dir_or_file, filter)
    if not name:
        return None

    # Update recently used dir.
    if isinstance(name, list):
        assert func == qw.QFileDialog.getOpenFileNames
        dir = os.path.dirname(name[0])
    else:
        assert isinstance(name, str)
        dir = os.path.dirname(name)

    history_dir.set(dir)
    return name


def get_open_file_name(
    parent: qw.QWidget,
    title: str,
    history_dir: _gp.Ref[_gp.GlobalPrefs],
    filters: List[str],
) -> Optional[str]:
    name = _get_hist_name(
        qw.QFileDialog.getOpenFileName, parent, title, history_dir, None, filters
    )
    if name:
        assert isinstance(name, str)
        return name
    return None


def get_open_file_list(
    parent: qw.QWidget,
    title: str,
    history_dir: _gp.Ref[_gp.GlobalPrefs],
    filters: List[str],
) -> Optional[List[str]]:
    name = _get_hist_name(
        qw.QFileDialog.getOpenFileNames, parent, title, history_dir, None, filters
    )
    if name:
        assert isinstance(name, list)
        return name
    return None


@attr.dataclass
class KeyFilter:
    key: Optional[str]
    filter: str


@attr.dataclass
class SaveName:
    name: str

    # Used to key into get_save_file_path(filters).
    suffix: Optional[str]


def get_save_file_path(
    parent: qw.QWidget,
    title: str,
    history_dir: _gp.Ref[_gp.GlobalPrefs],
    initial_stem: str,
    filters: Dict[Optional[str], str],
    suffix: Optional[str],
) -> Optional[SaveName]:
    """
    Given a working directory and initial filename, request and return a filename to
    save a file.

    This function takes multiple filetypes, an initial filetype (by string
    extension), and additionally returns the filetype the user selected.
    """

    init_filter = filters.get(suffix, None)
    # If unsupported save extension (newer version?), use default extension.
    if not init_filter:
        suffix, init_filter = next(iter(filters.items()))

    # Get initial directory for the dialog.
    init_path: str = history_dir.get()

    # Get initial filename if present.
    if initial_stem:
        init_path = os.path.join(init_path, os.path.basename(initial_stem))
        # Append file extension.
        if suffix:
            init_path += suffix

    # Get filename from dialog.
    filter_str = ";;".join(filters.values())
    out_name, sel_filter = qw.QFileDialog.getSaveFileName(
        parent, title, init_path, filter_str, init_filter
    )
    if not out_name:
        return None

    for suffix, filter in filters.items():
        if filter == sel_filter:
            out_suffix = suffix
            break
    else:
        # getSaveFileName() will always return a filter from the list (with whitespace collapsed,
        # https://github.com/qt/qtbase/blob/ec011141b8d17a2edc58e0d5b6ebb0f1632fff90/src/widgets/dialogs/qfiledialog.cpp#L1415).
        # This block is only reachable if the returned filter is not in `filter_str` passed in.
        # This indicates a bug in the caller (incorrect whitespace in filters.values()).
        raise RuntimeError(
            f"unrecognized file filter {repr(sel_filter)} in {list(filters.values())}"
        )

    user_suffix = Path(out_name).suffix

    # If user explicitly types a different extension, use it next time (even if they
    # don't change the dropdown). This works on Windows and FIXME Linux? Mac?
    # FIXME it seems confusing to pick a different default extension if you type but don't dropdown? IDK.
    if user_suffix and user_suffix in filters:
        out_suffix = user_suffix

    # Append suffix if missing in user-specified name but present in dialog.
    # getSaveFileName() does not automatically append a suffix on Linux KDE.
    if user_suffix == "" and out_suffix:
        out_name += out_suffix

    # Update recently used dir.
    dir = os.path.dirname(out_name)
    history_dir.set(dir)
    return SaveName(out_name, out_suffix)
