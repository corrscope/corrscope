import pickle
from io import StringIO, BytesIO
from typing import ClassVar, TYPE_CHECKING, Type, TypeVar

import attr
from ruamel.yaml import (
    yaml_object,
    YAML,
    Representer,
    RoundTripRepresenter,
    Constructor,
    Node,
)

if TYPE_CHECKING:
    from enum import Enum


__all__ = [
    "yaml",
    "copy_config",
    "register_config",
    "kw_config",
    "Alias",
    "Ignored",
    "register_enum",
    "TypedEnumDump",
    "CorrError",
    "CorrWarning",
]


# Setup YAML loading (yaml object).


class MyYAML(YAML):
    def dump(self, data, stream=None, **kw):
        """ Allow dumping to str. """
        inefficient = False
        if stream is None:
            inefficient = True
            stream = StringIO()
        YAML.dump(self, data, stream, **kw)
        if inefficient:
            return stream.getvalue()


class NoAliasRepresenter(RoundTripRepresenter):
    """ Disable aliases. """

    def ignore_aliases(self, data):
        return True


# Default typ='roundtrip' creates 'ruamel.yaml.comments.CommentedMap' instead of dict.
# Is isinstance(CommentedMap, dict)? IDK
yaml = MyYAML()
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
pickle is faster, but less general (works fine for @register_config objects).
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


def register_config(cls=None, *, kw_only=False, always_dump: str = ""):
    """ Marks class as attrs, and enables YAML dumping (excludes default fields). """

    def decorator(cls: Type):
        cls.__getstate__ = _ConfigMixin.__getstate__
        cls.__setstate__ = _ConfigMixin.__setstate__
        cls.always_dump = always_dump

        # https://stackoverflow.com/a/51497219/2683842
        # YAML().register_class(cls) works... on versions more recent than 2018-07-12.
        return _yaml_loadable(attr.dataclass(cls, kw_only=kw_only))

    if cls is not None:
        return decorator(cls)
    else:
        return decorator


def kw_config(*args, **kwargs):
    return register_config(*args, **kwargs, kw_only=True)


@attr.dataclass()
class _ConfigMixin:
    """
    Class is unused. __getstate__ and __setstate__ are assigned into other classes.
    Ideally I'd use inheritance, but @yaml_object and @dataclass rely on decorators,
    and I want @register_config to Just Work and not need inheritance.
    """

    always_dump: ClassVar[str]

    # SafeRepresenter.represent_yaml_object() uses __getstate__ to dump objects.
    def __getstate__(self):
        """ Removes all fields with default values, but not found in
        self.always_dump. """

        always_dump = set(self.always_dump.split())
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


@attr.dataclass
class Alias:
    """
    @register_config
    class Foo:
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


class TypedEnumDump:
    def __init_subclass__(cls, **kwargs):
        _yaml_loadable(cls)

    @classmethod
    def to_yaml(cls: Type["Enum"], representer: Representer, node: "Enum"):
        return representer.represent_scalar("!" + cls.__name__, node._name_)

    @classmethod
    def from_yaml(cls: Type["Enum"], constructor: Constructor, node: "Node"):
        return cls[node.value]


# Miscellaneous


class CorrError(ValueError):
    """ Error caused by invalid end-user input (via YAML/GUI config).
    (Should be) caught by GUI and displayed to user. """

    pass


class CorrWarning(UserWarning):
    """ Warning about deprecated end-user config (YAML/GUI).
    (Should be) caught by GUI and displayed to user. """

    pass
