import os
import platform
import sys
from pathlib import Path
from typing import MutableMapping, List

from appdirs import user_data_dir

import corrscope
from corrscope.config import CorrError


__all__ = ["appdata_dir", "PATH_dir", "get_ffmpeg_url", "MissingFFmpegError"]


def prepend(dic: MutableMapping[str, str], _key: List[str], prefix: str) -> None:
    """Dubiously readable syntactic sugar for prepending to a string in a dict."""
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

    if sys.platform == "win32" and is_os_64:
        return "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.7z"
    else:
        return ""


class MissingFFmpegError(CorrError, FileNotFoundError):
    ffmpeg_url = get_ffmpeg_url()
    can_download = bool(ffmpeg_url)

    message = (
        f'FFmpeg+FFplay must be in PATH or "{PATH_dir}" in order to use corrscope.\n'
    )

    if can_download:
        message += (
            f"Download ffmpeg from {ffmpeg_url}, "
            f"open in 7-Zip and navigate to the ffmpeg-.../bin folder, "
            f"and copy all .exe files to the folder above."
        )
    else:
        message += "Cannot download FFmpeg for your platform."

    def __str__(self) -> str:
        return self.message
