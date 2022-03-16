import os.path
from pathlib import Path
from typing import *

import qtpy.QtWidgets as qw
import attr

import corrscope.settings.global_prefs as _gp

Name = str
Filter = str


@attr.dataclass
class FileName:
    file_name: Optional[str]
    file_list: Optional[List[str]]
    sel_filter: str


def _get_hist_name(
    func: Callable[..., Tuple[str, str]],
    parent: qw.QWidget,
    title: str,
    history_dir: _gp.Ref[_gp.GlobalPrefs],
    default_name: Optional[str],
    filters: List[str],
) -> Optional[FileName]:
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
        history_dir.set(dir)
        return FileName(None, name, sel_filter)

    else:
        assert isinstance(name, str)
        dir = os.path.dirname(name)
        history_dir.set(dir)
        return FileName(name, None, sel_filter)


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
        assert name.file_name is not None
        return name.file_name
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
        assert name.file_list is not None
        return name.file_list
    return None


# Returns Path for legacy reasons. Others return str.
def get_save_file_path(
    parent: qw.QWidget,
    title: str,
    history_dir: _gp.Ref[_gp.GlobalPrefs],
    default_name: str,
    filters: List[str],
    default_suffix: str,
) -> Optional[Path]:
    name = _get_hist_name(
        qw.QFileDialog.getSaveFileName,
        parent,
        title,
        history_dir,
        default_name,
        filters,
    )
    if name:
        assert name.file_name is not None
        path = Path(name.file_name)

        if name.sel_filter == filters[0] and path.suffix == "":
            path = path.with_suffix(default_suffix)

        return path
    else:
        return None
