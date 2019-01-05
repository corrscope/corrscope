import os
import platform
import sys
from pathlib import Path

from corrscope.config import CorrError

# Add app-specific ffmpeg path.

path_dir = str(Path(__file__).parent / "path")
os.environ["PATH"] += os.pathsep + path_dir
# Editing sys.path doesn't work.
# https://bugs.python.org/issue8557 is relevant but may be outdated.


# Unused
# def ffmpeg_exists():
#     return shutil.which("ffmpeg") is not None


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
