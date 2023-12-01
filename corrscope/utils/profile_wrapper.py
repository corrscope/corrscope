import datetime
from itertools import count
from pathlib import Path
from typing import Iterator, Union, Callable, Any


__all__ = ["run_profile"]


PROFILE_DUMP_NAME = "yappi"


def get_profile_dump_name(prefix: str) -> str:
    now_date = datetime.datetime.now()
    now_str = now_date.strftime("%Y-%m-%d_T%H-%M-%S")

    # Pycharm can't load CProfile files with dots in the name.
    prefix = prefix.split(".")[0]

    profile_dump_name = f"callgrind.out.{prefix}-{PROFILE_DUMP_NAME}-{now_str}"

    # Write stats to unused filename
    for name in add_numeric_suffixes(profile_dump_name):
        abs_path = Path(name).resolve()
        if not abs_path.exists():
            return str(abs_path)
    assert False  # never happens since add_numeric_suffixes is endless.


def add_numeric_suffixes(s: str) -> Iterator[str]:
    """f('foo')
    yields 'foo', 'foo2', 'foo3'...
    """
    yield s
    for i in count(2):
        yield s + str(i)


def run_profile(command: Callable[[], Any], dump_prefix: Union[str, Path]):
    import yappi

    profile_dump_name = get_profile_dump_name(str(dump_prefix))

    yappi.start()
    try:
        command()
    finally:
        yappi.stop()

        prof = yappi.get_func_stats()
        prof.print_all()
        prof.save(profile_dump_name, "CALLGRIND")
