import pickle
from io import StringIO, BytesIO
from typing import ClassVar, TYPE_CHECKING, Type, TypeVar, Set

import attr
from ruamel.yaml import yaml_object, YAML, Representer

if TYPE_CHECKING:
    from enum import Enum


__all__ = [
    "yaml",
    "copy_config",
    "DumpableAttrs",
    "KeywordAttrs",
    "Alias",
    "Ignored",
    "register_enum",
    "CorrError",
    "CorrWarning",
]


# Setup YAML loading (yaml object).


class MyYAML(YAML):
    def dump(self, data, stream=None, **kw):
        inefficient = False
        if stream is None:
            inefficient = True
            stream = StringIO()
        YAML.dump(self, data, stream, **kw)
        if inefficient:
            return stream.getvalue()


# Default typ='roundtrip' creates 'ruamel.yaml.comments.CommentedMap' instead of dict.
# Is isinstance(CommentedMap, dict)? IDK
yaml = MyYAML()
_yaml_loadable = yaml_object(yaml)


"""
Speed of copying objects:

number = 100
print(timeit.timeit(lambda: f(cfg), number=number))

- pickle_copy 0.0566s
- deepcopy    0.0967s
- yaml_copy   0.4875s

pickle_copy is fastest.

According to https://stackoverflow.com/questions/1410615/ ,
pickle is faster, but less general (works fine for DumpableAttrs objects).
"""

T = TypeVar("T")

# Unused
# def yaml_copy(obj: T) -> T:
#     with StringIO() as stream:
#         yaml.dump(obj, stream)
#         return yaml.load(stream)

# AKA pickle_copy
def copy_config(obj: T) -> T:
    with BytesIO() as stream:
        pickle.dump(obj, stream)
        stream.seek(0)
        return pickle.load(stream)


# Setup configuration load/dump infrastructure.


class DumpableAttrs:
    """ Marks class as attrs, and enables YAML dumping (excludes default fields). """

    __always_dump: ClassVar[Set[str]]

    if TYPE_CHECKING:

        def __init__(self, *args, **kwargs):
            pass

    def __init_subclass__(cls, kw_only=False, always_dump: str = ""):
        cls.__always_dump = set(always_dump.split())
        del always_dump

        _yaml_loadable(attr.dataclass(cls, kw_only=kw_only))

        dump_fields = cls.__always_dump - {"*"}  # remove "*" if exists
        if "*" in cls.__always_dump:
            assert (
                not dump_fields
            ), f"Invalid always_dump, contains * and elements {dump_fields}"

        else:
            all_fields = {f.name for f in attr.fields(cls)}
            for dump_field in dump_fields:
                assert (
                    dump_field in all_fields
                ), f'Invalid always_dump="...{dump_field}" missing from class {cls.__name__}'

    # SafeRepresenter.represent_yaml_object() uses __getstate__ to dump objects.
    def __getstate__(self):
        """ Removes all fields with default values, but not found in
        self.always_dump. """

        always_dump = self.__always_dump
        dump_all = "*" in always_dump

        state = {}
        cls = type(self)

        for field in attr.fields(cls):
            # Skip deprecated fields with leading underscores.
            # They have already been baked into other config fields.

            name = field.name
            if name[0] == "_":
                continue

            value = getattr(self, name)

            if dump_all or name in always_dump:
                state[name] = value
                continue

            if field.default == value:
                continue
            # noinspection PyTypeChecker,PyUnresolvedReferences
            if (
                isinstance(field.default, attr.Factory)
                and field.default.factory() == value
            ):
                continue

            state[name] = value

        return state

    # SafeConstructor.construct_yaml_object() uses __setstate__ to load objects.
    def __setstate__(self, state):
        """ Redirect `Alias(key)=value` to `key=value`.
        Then call the dataclass constructor (to validate parameters). """

        for key, value in dict(state).items():
            class_var = getattr(self, key, None)

            if class_var is Ignored:
                del state[key]

            if isinstance(class_var, Alias):
                target = class_var.key
                if target in state:
                    raise CorrError(
                        f"{type(self).__name__} received both Alias {key} and "
                        f"equivalent {target}"
                    )

                state[target] = value
                del state[key]

        obj = type(self)(**state)
        self.__dict__ = obj.__dict__


class KeywordAttrs(DumpableAttrs):
    if TYPE_CHECKING:

        def __init__(self, **kwargs):
            pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(kw_only=True, **kwargs)


@attr.dataclass
class Alias:
    """
    class Foo(DumpableAttrs):
        x: int
        xx = Alias('x')     # do not add a type hint
    """

    key: str


Ignored = object()


# Setup Enum load/dump infrastructure


def register_enum(cls: Type):
    cls.to_yaml = _EnumMixin.to_yaml
    return _yaml_loadable(cls)


class _EnumMixin:
    @classmethod
    def to_yaml(cls, representer: Representer, node: "Enum"):
        return representer.represent_str(node._name_)


# Miscellaneous


class CorrError(ValueError):
    """ Error caused by invalid end-user input (via YAML/GUI config).
    (Should be) caught by GUI and displayed to user. """

    pass


class CorrWarning(UserWarning):
    """ Warning about deprecated end-user config (YAML/GUI).
    (Should be) caught by GUI and displayed to user. """

    pass
