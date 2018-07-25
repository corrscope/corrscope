import sys

from ruamel.yaml import yaml_object

from ovgenpy.config import register_config, yaml


def test_register_config():
    @register_config
    class Foo:
        foo: int
        bar: int

    yaml.dump(Foo(1, 2), sys.stdout)
    print()


def test_yaml_object():
    @yaml_object(yaml)
    class Bar:
        pass

    yaml.dump(Bar(), sys.stdout)
    print()
