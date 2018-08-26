# noinspection PyUnresolvedReferences
import sys

import pytest
from ruamel.yaml import yaml_object

from ovgenpy.config import register_config, yaml, Alias, Ignored

# YAML Idiosyncrasies: https://docs.saltstack.com/en/develop/topics/troubleshooting/yaml_idiosyncrasies.html

# Load/dump infrastructure testing
from ovgenpy.utils.keyword_dataclasses import fields


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


# Dataclass dump testing

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


# Dataclass load testing


def test_dump_load_aliases():
    """ Ensure dumping and loading `xx=Alias('x')` works.
    Ensure loading `{x=1, xx=1}` raises an error.
    Does not check constructor `Config(xx=1)`."""
    @register_config
    class Config:
        x: int
        xx = Alias('x')

    # Test dumping
    assert len(fields(Config)) == 1
    cfg = Config(1)
    s = yaml.dump(cfg)
    assert s == '''\
!Config
x: 1
'''
    assert yaml.load(s) == cfg

    # Test loading
    s = '''\
!Config
xx: 1
'''
    assert yaml.load(s) == Config(x=1)

    # Test exception on duplicated parameters.
    s = '''\
    !Config
    x: 1
    xx: 1
    '''
    with pytest.raises(TypeError):
        yaml.load(s)


def test_dump_load_ignored():
    """ Ensure loading `xx=Ignored` works.
    Does not check constructor `Config(xx=1)`.
    """
    @register_config
    class Config:
        xx = Ignored

    # Test dumping
    assert len(fields(Config)) == 0
    cfg = Config()
    s = yaml.dump(cfg)
    assert s == '''\
!Config {}
'''
    assert yaml.load(s) == cfg

    # Test loading
    s = '''\
!Config
xx: 1
'''
    assert yaml.load(s) == Config()


def test_load_argument_validation():
    """ Ensure that loading config via YAML catches missing and invalid parameters. """
    @register_config
    class Config:
        a: int

    yaml.load('''\
!Config
  a: 1
''')

    with pytest.raises(TypeError):
        yaml.load('!Config {}')

    with pytest.raises(TypeError):
        yaml.load('''\
!Config
  a: 1
  b: 1
''')


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
