import os
import platform
import sys
from typing import Dict, List
from pathlib import Path

from appdirs import user_data_dir

import corrscope
from corrscope.config import CorrError


__all__ = ["data_dir", "path_dir", "get_ffmpeg_url", "MissingFFmpegError"]


def prepend(dic: Dict[str, str], _key: List[str], prefix: str) -> None:
    """ Dubiously readable syntactic sugar for prepending to a string in a dict. """
    key = _key[0]
    dic[key] = prefix + dic[key]


data_dir = Path(user_data_dir(corrscope.app_name, appauthor=False, roaming=True))
data_dir.mkdir(exist_ok=True)

# Add app-specific ffmpeg path.
_path_dir = data_dir / "path"
_path_dir.mkdir(exist_ok=True)

path_dir = str(_path_dir)
prepend(os.environ, ["PATH"], path_dir + os.pathsep)
# Editing sys.path doesn't work.
# https://bugs.python.org/issue8557 is relevant but may be outdated.


def get_ffmpeg_url() -> str:
    # is_python_64 = sys.maxsize > 2 ** 32
    is_os_64 = platform.machine().endswith("64")

    def url(os_ver):
        return f"https://ffmpeg.zeranoe.com/builds/{os_ver}/shared/ffmpeg-latest-{os_ver}-shared.zip"

    if sys.platform == "win32" and is_os_64:
        return url("win64")
    elif sys.platform == "win32" and not is_os_64:
        return url("win32")
    elif sys.platform == "darwin" and is_os_64:
        return url("macos64")
    else:
        return ""


class MissingFFmpegError(CorrError):
    ffmpeg_url = get_ffmpeg_url()
    can_download = bool(ffmpeg_url)

    message = f'FFmpeg must be in PATH or "{path_dir}" in order to use corrscope.\n'

    if can_download:
        message += f"Download ffmpeg from {ffmpeg_url}."
    else:
        message += "Cannot download FFmpeg for your platform."

    def __str__(self):
        return self.message
