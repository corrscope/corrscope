from ruamel.yaml import yaml_object

from ovgenpy.config import register_config, yaml


def test_register_config():
    @register_config
    class Foo:
        foo: int
        bar: int

    s = yaml.dump(Foo(1, 2))
    assert s == '''\
!Foo
foo: 1
bar: 2
'''


def test_yaml_object():
    @yaml_object(yaml)
    class Bar:
        pass

    s = yaml.dump(Bar())
    assert s == '!Bar {}\n'


def test_exclude_defaults():
    @register_config
    class DefaultConfig:
        a: str = 'a'
        b: str = 'b'

    s = yaml.dump(DefaultConfig('alpha'))
    assert 'b:' not in s
    assert s == '''\
!DefaultConfig
a: alpha
'''
