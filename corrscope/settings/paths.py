import os
import platform
import sys
from typing import MutableMapping, List
from pathlib import Path

from appdirs import user_data_dir

import corrscope
from corrscope.config import CorrError


__all__ = ["appdata_dir", "PATH_dir", "get_ffmpeg_url", "MissingFFmpegError"]


def prepend(dic: MutableMapping[str, str], _key: List[str], prefix: str) -> None:
    """ Dubiously readable syntactic sugar for prepending to a string in a dict. """
    key = _key[0]
    dic[key] = prefix + dic[key]


appdata_dir = Path(user_data_dir(corrscope.app_name, appauthor=False, roaming=True))
appdata_dir.mkdir(parents=True, exist_ok=True)

# Add app-specific ffmpeg path.
_path_dir = appdata_dir / "path"
_path_dir.mkdir(exist_ok=True)

PATH_dir = str(_path_dir)
prepend(os.environ, ["PATH"], PATH_dir + os.pathsep)
# Editing sys.path doesn't work.
# https://bugs.python.org/issue8557 is relevant but may be outdated.


def get_ffmpeg_url() -> str:
    # is_python_64 = sys.maxsize > 2 ** 32
    is_os_64 = platform.machine().endswith("64")

    def url(os_ver: str) -> str:
        return f"https://ffmpeg.zeranoe.com/builds/{os_ver}/shared/ffmpeg-latest-{os_ver}-shared.zip"

    if sys.platform == "win32" and is_os_64:
        return url("win64")
    elif sys.platform == "win32" and not is_os_64:
        return url("win32")
    elif sys.platform == "darwin" and is_os_64:
        return url("macos64")
    else:
        return ""


class MissingFFmpegError(CorrError, FileNotFoundError):
    ffmpeg_url = get_ffmpeg_url()
    can_download = bool(ffmpeg_url)

    message = (
        f'FFmpeg+FFplay must be in PATH or "{PATH_dir}" in order to use corrscope.\n'
    )

    if can_download:
        message += f"Download ffmpeg from {ffmpeg_url}."
    else:
        message += "Cannot download FFmpeg for your platform."

    def __str__(self) -> str:
        return self.message
