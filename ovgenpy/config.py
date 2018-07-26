from io import StringIO

from dataclasses import dataclass, fields
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


class OvgenError(Exception):
    pass


def __getstate__(self):
    """ Returns all non-default fields. """
    state = {}
    cls = type(self)

    for field in fields(self):
        name = field.name
        value = getattr(self, name)
        default = getattr(cls, name, object())

        if value != default:
            state[name] = value

    return state


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


def register_config(cls):
    """ Marks class as @dataclass, and enables YAML dumping (excludes default fields).

    dataclasses.dataclass is compatible with yaml.register_class.
    typing.NamedTuple is incompatible.
    """

    # SafeRepresenter.represent_yaml_object() uses __getstate__ to dump objects.
    cls.__getstate__ = __getstate__
    # SafeConstructor.construct_yaml_object() uses __setstate__ to load objects.
    cls.__setstate__ = __setstate__

    # https://stackoverflow.com/a/51497219/2683842
    # YAML().register_class(cls) works... on versions more recent than 2018-07-12.
    return yaml_object(yaml)(
        dataclass(cls)
    )


# __init__-less non-dataclasses are also compatible with yaml.register_class.
