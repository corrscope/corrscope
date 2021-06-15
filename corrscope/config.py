"""
The most important class in this module is `DumpableAttrs`.
See its docstring for details.
"""

import pickle
import warnings
from enum import Enum
from io import StringIO, BytesIO
from pathlib import Path
from typing import (
    ClassVar,
    TypeVar,
    Type,
    FrozenSet,
    Optional,
    TYPE_CHECKING,
    Dict,
    Any,
    TextIO,
    Union,
    IO,
)

import attr
from ruamel.yaml import (
    yaml_object,
    YAML,
    Representer,
    RoundTripRepresenter,
    Constructor,
    Node,
)

__all__ = [
    "yaml",
    "copy_config",
    "DumpableAttrs",
    "evolve_compat",
    "KeywordAttrs",
    "with_units",
    "get_units",
    "Alias",
    "Ignored",
    "DumpEnumAsStr",
    "TypedEnumDump",
    "CorrError",
    "CorrWarning",
]


# Setup YAML loading (yaml object).


class MyYAML(YAML):
    def dump(
        self, data: Any, stream: "Union[Path, TextIO, None]" = None, **kwargs
    ) -> Optional[str]:

        # On Windows, when dumping to path, ruamel.yaml writes files in locale encoding.
        # Foreign characters are undumpable. Locale-compatible characters cannot be loaded.
        # https://bitbucket.org/ruamel/yaml/issues/316/unicode-encoding-decoding-errors-on
        # Both are bad, so use UTF-8.
        if isinstance(stream, Path):
            with stream.open("w", encoding="utf-8") as f:
                self.dump_without_corrupting(data, f, **kwargs)

        elif stream is None:
            # Possibly only called in unit tests, not in production.
            stream = StringIO()
            self.dump_without_corrupting(data, stream, **kwargs)
            return stream.getvalue()

        else:
            # with atomic_write(...) as f: dump(..., f)
            self.dump_without_corrupting(data, stream, **kwargs)

    def dump_without_corrupting(self, *args, **kwargs):
        YAML.dump(self, *args, **kwargs)

    def load(self, stream):
        """
        If a file was dumped by a prior version of corrscope using locale encoding,
        and contains non-ASCII characters,
        ruamel.yaml cannot guess the encoding properly (out of UTF-8, 16, or 32).
        We should instead try loading using the locale.
        https://bitbucket.org/ruamel/yaml/issues/316/unicode-encoding-decoding-errors-on
        """
        if isinstance(stream, Path):
            try:
                with stream.open("r", encoding="utf-8") as f:
                    s = f.read()
            except UnicodeDecodeError:
                with stream.open("r") as f:
                    s = f.read()

            return self.load_without_corrupting(s)

        elif isinstance(stream, str):
            return self.load_without_corrupting(stream)

        else:
            raise TypeError

    def load_without_corrupting(self, *args, **kwargs):
        return YAML.load(self, *args, **kwargs)


class NoAliasRepresenter(RoundTripRepresenter):
    """
    Ensure that dumping 2 identical enum values
    doesn't produce ugly aliases.
    TODO test
    """

    def ignore_aliases(self, data: Any) -> bool:
        if isinstance(data, Enum):
            return True
        return super().ignore_aliases(data)


yaml = MyYAML()
yaml.width = float("inf")

# Default typ='roundtrip' creates 'ruamel.yaml.comments.CommentedMap' instead of dict.
# Is isinstance(CommentedMap, dict)? IDK
assert yaml.Representer == RoundTripRepresenter
yaml.Representer = NoAliasRepresenter

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
    """Marks class as attrs, and enables YAML dumping (excludes default fields).

    It is subclassed for
    (statically-typed key-value objects which will be dumped/loaded as YAML files).

    ## Subclassing `DumpableAttrs` and converting to YAML

    - This class works like `dataclasses` or `attrs.dataclass` decorators,
      and wraps the latter.
    - Subclass `DumpableAttrs`, then add class-level type annotations.
      These annotations are converted into `__init__(...)` constructor parameters.

    ```py
    class Config(DumpableAttrs):
        path: str
        priority: int = 0
    ```

    Unlike many other libraries and usual JSON,
    the YAML string representation of a `DumpableAttrs` object
    encodes what Python type the object is.
    For example, Config("foo.wav", 1) is dumped as:

    ```yaml
    !Config
    path: foo.wav  # Required
    priority: 1  # If not present, defaults to 0
    # See DumpableAttrs docstring for details.
    ```

    ## Polymorphism

    The YAML file determines what type is loaded,
    so the config type can be used to pick a object type at runtime.

    - For example, putting `!CorrelationTriggerConfig` in YAML
      loads a `CorrelationTriggerConfig` config object,
      which tells corrscope to create a `CorrelationTrigger` algorithm object.
    - Putting `!NullTriggerConfig` in YAML
      instead loads a `NullTriggerConfig` config object,
      which tells corrscope to create a `NullTrigger` algorithm object.
      (This is only used for unit tests, and is incompatible with GUI.)

    ## `KeywordAttrs` are similar, with 2 differences:

    - Subclasses can have non-default arguments after default arguments.
    - Its constructor can only be called with keyword (a=1, b=2) arguments,
      not positional (1, 2).

    ## Optional Parameters

    class Config(DumpableAttrs, always_dump="", exclude=""): ...
    - `always_dump` contains whitespace-separated list of fields to always dump
      (if equal to default).
    - If always_dump="*", `exclude` contains whitespace-separated list of fields
      to not dump (if equal to default).

    ## Loading from YAML: Alias and Ignored

    YAML loading uses __getstate__ and __setstate__.

    If `old = Alias("new")`,
    then loading a YAML file `old: value` initializes `new = value`.

    If `old = Ignored`,
    then loading a YAML file `old: value` silently discards `value`.
    Ignored is unused in my code,
    since __setstate__ automatically discards unrecognized fields (with a warning).
    """

    if TYPE_CHECKING:

        def __init__(self, *args, **kwargs):
            pass

    # Private variable, to avoid clashing with subclass attributes.
    __always_dump: ClassVar[FrozenSet[str]] = frozenset()
    __exclude: ClassVar[FrozenSet[str]] = frozenset()

    def __init_subclass__(
        cls, kw_only: bool = False, always_dump: str = "", exclude: str = ""
    ):
        _yaml_loadable(attr.dataclass(cls, kw_only=kw_only))

        # Merge always_dump with superclass's __always_dump.
        super_always_dump = cls.__always_dump
        super_exclude = cls.__exclude
        assert type(super_always_dump) == frozenset
        assert type(super_exclude) == frozenset

        cls.__always_dump = super_always_dump | frozenset(always_dump.split())
        cls.__exclude = super_exclude | frozenset(exclude.split())

        del super_always_dump, always_dump
        del super_exclude, exclude

        all_fields = {f.name for f in attr.fields(cls)}
        dump_fields = cls.__always_dump - {"*"}  # remove "*" if exists
        exclude_fields = cls.__exclude

        if "*" in cls.__always_dump:
            assert (
                not dump_fields
            ), f"Invalid always_dump, contains * and elements {dump_fields}"

            for exclude_field in exclude_fields:
                assert (
                    exclude_field in all_fields
                ), f'Invalid exclude, contains "{exclude_field}" missing from class {cls.__name__}'

        else:
            assert (
                not exclude_fields
            ), f"Invalid exclude, always_dump does not contain *"

            for dump_field in dump_fields:
                assert (
                    dump_field in all_fields
                ), f'Invalid always_dump, contains "{dump_field}" missing from class {cls.__name__}'

    # SafeRepresenter.represent_yaml_object() uses __getstate__ to dump objects.
    def __getstate__(self) -> Dict[str, Any]:
        """Removes all fields with default values, but not found in
        self.always_dump."""

        always_dump = self.__always_dump
        dump_all = "*" in always_dump
        exclude = self.__exclude

        state = {}
        cls = type(self)

        def should_dump(attr_name, value) -> bool:
            """
            Sure it would be simpler to dump and load __dict__ directly,
            and not deal with __init__ underscore stripping,
            but I'd lose structure checking, converters, and __attrs_post_init__.
            """

            # Dump values marked as always dumped.
            if attr_name in always_dump:
                return True

            if dump_all and attr_name not in exclude:
                return True

            # Don't dump default values.
            if field.default == value:
                return False
            # noinspection PyTypeChecker,PyUnresolvedReferences
            if (
                isinstance(field.default, attr.Factory)  # type: ignore
                and field.default.factory() == value  # type: ignore
            ):
                return False

            # Dump values with different or missing defaults.
            return True

        for field in attr.fields(cls):
            # Used for getattr(), dot access.
            attr_name = field.name

            # Dumped to state, and passed into __init__.
            state_name = attr_name
            if state_name[0] == "_":
                state_name = state_name[1:]

            # Skip fields which cannot be passed into __init__().
            # - init=False
            if not field.init:
                continue

            value = getattr(self, attr_name)
            if should_dump(attr_name, value):
                state[state_name] = value

        return state

    # SafeConstructor.construct_yaml_object() uses __setstate__ to load objects.
    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__ = self.new_from_state(state).__dict__

    # If called via instance, cls == type(self).
    @classmethod
    def new_from_state(cls: Type[T], state: Dict[str, Any]) -> T:
        """Redirect `Alias(key)=value` to `key=value`.
        Then call the dataclass constructor (to validate parameters)."""

        cls_name = cls.__name__
        fields = attr.fields_dict(cls)

        # All names which can be passed into __init__()
        field_names = {name.lstrip("_") for name, field in fields.items() if field.init}

        new_state = {}
        for key, value in dict(state).items():
            class_var = getattr(cls, key, None)

            if class_var is Ignored:
                pass

            elif isinstance(class_var, Alias):
                target = class_var.key
                if target in state:
                    raise CorrError(
                        f"{cls_name} received both Alias {key} and "
                        f"equivalent {target}"
                    )
                new_state[target] = value

            elif key not in field_names:
                warnings.warn(
                    f'Unrecognized field "{key}" in !{cls_name}, ignoring', CorrWarning
                )

            else:
                new_state[key] = value

        del state
        return cls(**new_state)


def evolve_compat(obj: DumpableAttrs, **changes):
    """Evolve an object, based on user-specified dict,
    while ignoring unrecognized keywords."""
    # In dictionaries, later values will always override earlier ones
    return obj.new_from_state({**obj.__dict__, **changes})


class KeywordAttrs(DumpableAttrs):
    if TYPE_CHECKING:

        def __init__(self, **kwargs):
            pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(kw_only=True, **kwargs)


UNIT_SUFFIX = "suffix"


def with_units(unit, **kwargs):
    metadata = {UNIT_SUFFIX: f" {unit}"}
    return attr.ib(metadata=metadata, **kwargs)


def get_units(field: attr.Attribute) -> str:
    return field.metadata.get(UNIT_SUFFIX, "")


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
SomeEnum = TypeVar("SomeEnum", bound=Enum)


@classmethod
def _by_name(cls: Type[SomeEnum], enum_or_name: Union[SomeEnum, str]) -> SomeEnum:
    if isinstance(enum_or_name, cls):
        return enum_or_name
    else:
        try:
            return cls[enum_or_name]
        except KeyError:
            raise CorrError(
                f"invalid {cls.__name__} '{enum_or_name}' not in "
                f"{[el.name for el in cls]}"
            )


class DumpEnumAsStr(Enum):
    def __init_subclass__(cls):
        _yaml_loadable(cls)

    @classmethod
    def to_yaml(cls, representer: Representer, node: Enum) -> Any:
        return representer.represent_str(node._name_)  # type: ignore

    by_name = _by_name


class TypedEnumDump(Enum):
    def __init_subclass__(cls):
        _yaml_loadable(cls)

    @classmethod
    def to_yaml(cls, representer: Representer, node: Enum) -> Any:
        return representer.represent_scalar(
            "!" + cls.__name__, node._name_  # type: ignore
        )

    @classmethod
    def from_yaml(cls, constructor: Constructor, node: Node) -> Enum:
        return cls[node.value]

    by_name = _by_name


# Miscellaneous


class CorrError(ValueError):
    """Error caused by invalid end-user input (via YAML/GUI config).
    (Should be) caught by GUI and displayed to user."""

    pass


class CorrWarning(UserWarning):
    """Warning about deprecated end-user config (YAML/GUI).
    (Should be) caught by GUI and displayed to user."""

    pass
