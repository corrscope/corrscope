from dataclasses import dataclass
from ruamel.yaml import YAML, yaml_object


# typing.NamedTuple is incompatible with yaml.register_class.
# dataclasses.dataclass is compatible.
# So use the latter.

# __init__-less classes are also compatible with yaml.register_class.


yaml = YAML()


def register_dataclass(cls):
    # https://stackoverflow.com/a/51497219/2683842
    # YAML.register_class(cls) has only returned cls since 2018-07-12.
    return yaml_object(yaml)(
        dataclass(cls)
    )
