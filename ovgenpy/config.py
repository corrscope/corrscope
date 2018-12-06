from io import StringIO
from typing import ClassVar, TYPE_CHECKING, Type

import attr
from ruamel.yaml import yaml_object, YAML, Representer

if TYPE_CHECKING:
    from enum import Enum


__all__ = ['yaml',
           'register_config', 'kw_config', 'Alias', 'Ignored',
           'register_enum', 'OvgenError']


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


# Setup configuration load/dump infrastructure.

def register_config(cls=None, *, kw_only=False, always_dump: str = ''):
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
        dump_all = ('*' in always_dump)

        state = {}
        cls = type(self)

        for field in attr.fields(cls):
            name = field.name
            value = getattr(self, name)

            if dump_all or name in always_dump:
                state[name] = value
                continue

            if field.default == value:
                continue
            # noinspection PyTypeChecker,PyUnresolvedReferences
            if isinstance(field.default, attr.Factory) \
                    and field.default.factory() == value:
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
                    raise TypeError(
                        f'{type(self).__name__} received both Alias {key} and '
                        f'equivalent {target}'
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
    def to_yaml(cls, representer: Representer, node: 'Enum'):
        return representer.represent_str(node._name_)


# Miscellaneous

class OvgenError(ValueError):
    """ Error caused by invalid end-user input (via CLI or YAML config). """
    pass


