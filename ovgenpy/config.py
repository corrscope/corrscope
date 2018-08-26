from io import StringIO
from typing import ClassVar, TYPE_CHECKING

from ovgenpy.utils.keyword_dataclasses import dataclass, fields, field
# from dataclasses import dataclass, fields
from ruamel.yaml import yaml_object, YAML, Representer

if TYPE_CHECKING:
    from enum import Enum

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


# https://yaml.readthedocs.io/en/latest/dumpcls.html
# >Only yaml = YAML(typ='unsafe') loads and dumps Python objects out-of-the-box. And
# >since it loads any Python object, this can be unsafe.
# I assume roundtrip is safe.
yaml = MyYAML()
_yaml_loadable = yaml_object(yaml)


# Setup configuration load/dump infrastructure.

def register_config(cls=None, *, always_dump: str = ''):
    """ Marks class as @dataclass, and enables YAML dumping (excludes default fields).

    dataclasses.dataclass is compatible with yaml_object().
    typing.NamedTuple is incompatible.
    """

    def decorator(cls: type):
        cls.__getstate__ = _ConfigMixin.__getstate__
        cls.__setstate__ = _ConfigMixin.__setstate__
        cls.always_dump = always_dump

        # https://stackoverflow.com/a/51497219/2683842
        # YAML().register_class(cls) works... on versions more recent than 2018-07-12.
        return _yaml_loadable(dataclass(cls))

    if cls is not None:
        return decorator(cls)
    else:
        return decorator


@dataclass()
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
        dump_all = ('*' in always_dump)

        state = {}
        cls = type(self)

        for field in fields(self):
            name = field.name
            value = getattr(self, name)

            if dump_all or name in always_dump:
                state[name] = value
                continue

            default = getattr(cls, name, object())
            if value != default:
                state[name] = value

        return state

    # SafeConstructor.construct_yaml_object() uses __setstate__ to load objects.
    def __setstate__(self, state):
        """ Redirect `Alias(key)=value` to `key=value`.
        Then call the dataclass constructor (to validate parameters). """

        for key, value in dict(state).items():
            classvar = getattr(self, key, None)
            if not isinstance(classvar, Alias):
                continue

            target = classvar.key
            if target in state:
                raise TypeError(
                    f'{type(self).__name__} received both Alias {key} and equivalent '
                    f'{target}'
                )

            state[target] = value
            del state[key]

        obj = type(self)(**state)
        self.__dict__ = obj.__dict__


@dataclass
class Alias:
    """
    @register_config
    class Foo:
        x: int
        xx = Alias('x')     # do not add a type hint
    """
    key: str


# Unused
def default(value):
    """Supplies a mutable default value for a dataclass field."""
    string = repr(value)
    return field(default=lambda: eval(string))


# Setup Enum load/dump infrastructure

def register_enum(cls: type):
    cls.to_yaml = _EnumMixin.to_yaml
    return _yaml_loadable(cls)


class _EnumMixin:
    @classmethod
    def to_yaml(cls, representer: Representer, node: 'Enum'):
        return representer.represent_str(node._name_)


# Miscellaneous

class OvgenError(Exception):
    pass


