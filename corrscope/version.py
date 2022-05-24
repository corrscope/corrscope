import os
import platform
import struct
import sys
from pathlib import Path


_package = Path(__file__).parent


# Version prefix
base_version = "0.8.1"
is_dev = "-" in base_version


def _base_plus_metadata(build_metadata: str) -> str:
    return base_version + "+" + build_metadata


# Human-readable after downloading .zip.
version_txt: Path = _package / "version.txt"

# Build metadata: accessed by the program.
# https://semver.org/#spec-item-10
version_py: Path = _package / "_version.py"
metadata_key = "build_metadata"


def get_version() -> str:
    """Called at runtime (maybe pyinstaller time too).
    Depends on pyinstaller_write_version and filesystem.
    """
    is_installer = hasattr(sys, "frozen")

    # PyInstaller's .spec file creates _version.py and version.txt.
    if is_installer and is_dev:
        import corrscope._version

        build_metadata = getattr(corrscope._version, metadata_key)
        return _base_plus_metadata(build_metadata)
    else:
        return base_version


def pyinstaller_write_version() -> str:
    """Returns version.

    Called only at pyinstaller time.
    Writes to filesystem, does NOT call get_version().
    Filesystem is ignored if version number isn't prerelease (x.y.z-pre).
    """
    if is_dev:
        build_metadata = _calc_metadata()
        version = _base_plus_metadata(build_metadata)
    else:
        build_metadata = ""
        version = base_version

    os = platform.system().lower()
    if os == "windows":
        os = "win"

    # 32 or 64 bit
    arch = str(struct.calcsize("P") * 8)

    version = f"{version}-{os}{arch}"

    with version_txt.open("w") as txt:
        txt.write(version)

    with version_py.open("w") as f:
        f.write(f"{metadata_key} = {repr(build_metadata)}")

    return version


# Compute version suffix
env = {}


def alias_env(new: str, old: str) -> str:
    if old in os.environ:
        env[new] = os.environ[old]
    return new


is_appveyor = "APPVEYOR" in os.environ
if is_appveyor:
    BRANCH = alias_env("BRANCH", "APPVEYOR_REPO_BRANCH")
    PR_NUM = alias_env("PR_NUM", "APPVEYOR_PULL_REQUEST_NUMBER")
    PR_BRANCH = alias_env("PR_BRANCH", "APPVEYOR_PULL_REQUEST_HEAD_REPO_BRANCH")

    # "buildN" where N=APPVEYOR_BUILD_NUMBER
    VER = alias_env("VER", "APPVEYOR_BUILD_VERSION")


def _calc_metadata() -> str:
    """
    Build metadata MAY be denoted by appending a plus sign
    and a series of dot separated identifiers
    immediately following the patch or pre-release version.

    Identifiers MUST comprise only ASCII alphanumerics and hyphen [0-9A-Za-z-].
    """

    if not is_appveyor:
        return "local-build"

    is_pr = PR_NUM in env
    assert (PR_NUM in env) == (PR_BRANCH in env)

    assert VER in env

    if is_pr:
        return "{VER}.pr{PR_NUM}-{PR_BRANCH}".format(**env)
    else:
        if env[BRANCH] != "master":
            # Shouldn't happen, since side branches are not built.
            return "{VER}.{BRANCH}".format(**env)
        else:
            return "{VER}".format(**env)
