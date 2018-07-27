from io import StringIO
from typing import ClassVar

from ovgenpy.utils.keyword_dataclasses import dataclass, fields
from ruamel.yaml import yaml_object, YAML


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


def register_config(cls=None, *, always_dump: str = ''):
    """ Marks class as @dataclass, and enables YAML dumping (excludes default fields).

    dataclasses.dataclass is compatible with yaml.register_class.
    typing.NamedTuple is incompatible.
    """

    def decorator(cls: type):
        cls.__getstate__ = _ConfigMixin.__getstate__
        cls.__setstate__ = _ConfigMixin.__setstate__
        cls.always_dump = always_dump

        # https://stackoverflow.com/a/51497219/2683842
        # YAML().register_class(cls) works... on versions more recent than 2018-07-12.
        return yaml_object(yaml)(
            dataclass(cls)
        )

    if cls is not None:
        return decorator(cls)
    else:
        return decorator

    # __init__-less non-dataclasses are also compatible with yaml.register_class.


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
        """ Returns all fields with non-default value, or appeear in
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
        """ Checks that all fields match their correct types. """
        self.__dict__.update(state)
        for field in fields(self):
            key = field.name
            value = getattr(self, key)
            typ = field.type

            if not isinstance(value, typ):
                name = type(self).__name__
                raise OvgenError(f'{name}.{key} was supplied {repr(value)}, should be of type {typ.__name__}')

        if hasattr(self, '__post_init__'):
            self.__post_init__()


class OvgenError(Exception):
    pass


