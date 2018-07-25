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


def __getstate__(self):
    state = {}
    cls = type(self)

    for field in fields(self):
        name = field.name
        value = getattr(self, name)
        default = getattr(cls, name, object())

        if value != default:
            state[name] = value

    return state


def register_config(cls):
    """ Marks class as @dataclass, and enables YAML dumping (excludes default fields).

    dataclasses.dataclass is compatible with yaml.register_class.
    typing.NamedTuple is incompatible.
    """

    # SafeRepresenter.represent_yaml_object() uses __getstate__ to dump objects.
    cls.__getstate__ = __getstate__

    # https://stackoverflow.com/a/51497219/2683842
    # YAML().register_class(cls) works... on versions more recent than 2018-07-12.
    return yaml_object(yaml)(
        dataclass(cls)
    )


# __init__-less non-dataclasses are also compatible with yaml.register_class.
