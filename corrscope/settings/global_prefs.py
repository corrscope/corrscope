from typing import *

import attr
from corrscope.config import DumpableAttrs, yaml
from corrscope.settings import paths

# Default output extension.
VIDEO_NAME = ".mp4"


Attrs = TypeVar("Attrs", bound=DumpableAttrs)


class Ref(Generic[Attrs]):
    def __init__(self, obj: Attrs, key: str):
        self.obj = obj
        self.key = key

    def get(self) -> str:
        return getattr(self.obj, self.key)

    def set(self, value: str) -> None:
        setattr(self.obj, self.key, value)


@attr.dataclass
class Parallelism:
    parallel: bool = True
    max_render_cores: int = 2


class GlobalPrefs(DumpableAttrs, always_dump="*"):
    # Most recent YAML or audio file opened
    file_dir: str = ""

    @property
    def file_dir_ref(self) -> "Ref[GlobalPrefs]":
        return Ref(self, "file_dir")

    # Most recent video rendered
    separate_render_dir: bool = False
    render_dir: str = ""  # Set to "" whenever separate_render_dir=False.

    # Most recent video filetype, including period.
    render_ext: Optional[str] = VIDEO_NAME

    @property
    def render_dir_ref(self) -> "Ref[GlobalPrefs]":
        if self.separate_render_dir:
            return Ref(self, "render_dir")
        else:
            return self.file_dir_ref

    parallel: bool = True
    max_render_cores: int = 2

    def parallelism(self) -> Parallelism:
        return Parallelism(self.parallel, self.max_render_cores)


_PREF_PATH = paths.appdata_dir / "prefs.yaml"


def load_prefs() -> GlobalPrefs:
    try:
        pref = yaml.load(_PREF_PATH)
        if not isinstance(pref, GlobalPrefs):
            raise TypeError(f"prefs.yaml contains wrong type {type(pref)}")
        return pref

    except FileNotFoundError:
        return GlobalPrefs()


def dump_prefs(pref: GlobalPrefs) -> None:
    yaml.dump(pref, _PREF_PATH)
