import glob
import os
import shutil
import subprocess
from pathlib import Path

from PyInstaller.building.api import PYZ, EXE, COLLECT
from PyInstaller.building.build_main import Analysis
from PyInstaller.building.datastruct import TOC

from corrscope import version as v

block_cipher = None


def keep(dir, wildcard):
    includes = glob.glob(f"{dir}/{wildcard}")
    assert includes, f"{dir}, {wildcard} has no matches"
    return [(include, dir) for include in includes]


datas = keep("corrscope/gui", "*.ui") + keep("corrscope/path", "*")

version = v.pyinstaller_write_version()
datas.append((v.version_txt, "."))

app_name = "corrscope"
app_name_version = f"{app_name}-{version}"

a = Analysis(
    ["corrscope/__main__.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=["corrscope.gui.__init__"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["FixTk", "tcl", "tk", "_tkinter", "tkinter", "Tkinter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)


# Some dirs are included by PyInstaller hooks and must be removed after the fact.
_path_excludes = (
    # Matplotlib
    """
    mpl-data/fonts
    mpl-data/images
    mpl-data/sample_data
    """
    # PyQt
    """
    Qt5DBus.dll
    Qt5Network.dll
    Qt5Quick.dll
    Qt5Qml.dll
    Qt5Svg.dll
    Qt5WebSockets.dll
    """
    # QML file list taken from https://github.com/pyinstaller/pyinstaller/blob/0f31b35fe96de59e1a6faf692340a9ef93492472/PyInstaller/hooks/hook-PyQt5.py#L55
    """
    libEGL.dll libGLESv2.dll d3dcompiler_ opengl32sw.dll
    """
).split()
path_excludes = {s.lower() for s in _path_excludes}


def path_contains(path: str) -> bool:
    path = path.replace("\\", "/").lower()
    ret = any(x in path for x in path_excludes)
    return ret


# A TOC appears to be a list of tuples of the form (name, path, typecode).
# In fact, it's an ordered set, not a list.
# A TOC contains no duplicates, where uniqueness is based on name only.
def strip(arr: TOC):
    arr[:] = [
        (name, path, typecode)
        for (name, path, typecode) in arr
        if not path_contains(path)
    ]


strip(a.datas)
strip(a.binaries)


pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)


class ZipCollect(COLLECT):
    name: str  # dist/dir-name, != __init__(name=)

    def assemble(self):
        ret = super().assemble()

        if shutil.which("7z"):
            cmd_7z = "7z a -mx=3".split()

            # Don't use Path.with_suffix(), it removes trailing semver components.
            out_name = str(Path(self.name).with_name(app_name_version)) + ".7z"
            in_files = f"{self.name}/*"  # asterisk removes root directory from archive

            subprocess.run(cmd_7z + [out_name, in_files], check=True)
            assert os.path.getsize(out_name) > 2 ** 20

        return ret


coll = ZipCollect(
    exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=True, name=app_name
)
