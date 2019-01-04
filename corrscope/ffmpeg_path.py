import os
import shutil
from pathlib import Path

from corrscope.config import CorrError

# Add app-specific ffmpeg path.

path_dir = str(Path(__file__).parent / "path")
os.environ["PATH"] += os.pathsep + path_dir
# Editing sys.path doesn't work.
# https://bugs.python.org/issue8557 is relevant but may be outdated.


# Unused
def ffmpeg_exists():
    return shutil.which("ffmpeg") is not None


class MissingFFmpegError(CorrError):
    pass
