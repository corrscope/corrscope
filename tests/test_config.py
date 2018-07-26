# noinspection PyUnresolvedReferences
import sys

from dataclasses import fields
import pytest
from ruamel.yaml import yaml_object

from ovgenpy.config import register_config, yaml, OvgenError


# YAML Idiosyncrasies: https://docs.saltstack.com/en/develop/topics/troubleshooting/yaml_idiosyncrasies.html


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


def test_dump_defaults():
    @register_config
    class Config:
        a: str = 'a'
        b: str = 'b'

    s = yaml.dump(Config('alpha'))
    assert s == '''\
!Config
a: alpha
'''

    @register_config(always_dump='a b')
    class Config:
        a: str = 'a'
        b: str = 'b'
        c: str = 'c'

    s = yaml.dump(Config())
    assert s == '''\
!Config
a: a
b: b
'''

    @register_config(always_dump='*')
    class Config:
        a: str = 'a'
        b: str = 'b'

    s = yaml.dump(Config())
    assert s == '''\
!Config
a: a
b: b
'''


def test_load_type_checking():
    @register_config
    class Foo:
        foo: int
        bar: int

    s = '''\
!Foo
foo: "foo"
bar: "bar"
'''
    with pytest.raises(OvgenError) as e:
        print(yaml.load(s))
    print(e)


def test_load_post_init():
    """ yaml.load() does not natively call __post_init__. So @register_config modifies
    __setstate__ to call __post_init__. """
    @register_config
    class Foo:
        foo: int

        def __post_init__(self):
            self.foo = 99

    s = '''\
!Foo
foo: 0
'''
    assert yaml.load(s) == Foo(99)
