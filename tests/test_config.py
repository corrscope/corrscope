import pytest
from ruamel.yaml import yaml_object

from corrscope.config import (
    yaml,
    DumpableAttrs,
    KeywordAttrs,
    Alias,
    Ignored,
    CorrError,
)

# YAML Idiosyncrasies: https://docs.saltstack.com/en/develop/topics/troubleshooting/yaml_idiosyncrasies.html

# Load/dump infrastructure testing
import attr


def test_dumpable_attrs():
    class Foo(DumpableAttrs):
        foo: int
        bar: int

    s = yaml.dump(Foo(foo=1, bar=2))
    assert (
        s
        == """\
!Foo
foo: 1
bar: 2
"""
    )


def test_kw_config():
    class Foo(KeywordAttrs):
        foo: int = 1
        bar: int

    obj = Foo(bar=2)
    assert obj.foo == 1
    assert obj.bar == 2


def test_yaml_object():
    @yaml_object(yaml)
    class Bar:
        pass

    s = yaml.dump(Bar())
    assert s == "!Bar {}\n"


# Dataclass dump testing


def test_dump_defaults():
    class Config(DumpableAttrs):
        a: str = "a"
        b: str = "b"

    s = yaml.dump(Config("alpha"))
    assert (
        s
        == """\
!Config
a: alpha
"""
    )

    class Config(DumpableAttrs, always_dump="a b"):
        a: str = "a"
        b: str = "b"
        c: str = "c"

    s = yaml.dump(Config())
    assert (
        s
        == """\
!Config
a: a
b: b
"""
    )

    class Config(DumpableAttrs, always_dump="*"):
        a: str = "a"
        b: str = "b"

    s = yaml.dump(Config())
    assert (
        s
        == """\
!Config
a: a
b: b
"""
    )


def test_dump_default_factory():
    """ Ensure default factories are not dumped, unless attribute present
    in `always_dump`.

    Based on `attrs.Factory`. """

    class Config(DumpableAttrs):
        # Equivalent to attr.ib(factory=str)
        # See https://www.attrs.org/en/stable/types.html
        a: str = attr.Factory(str)
        b: str = attr.Factory(str)

    s = yaml.dump(Config("alpha"))
    assert (
        s
        == """\
!Config
a: alpha
"""
    )

    class Config(DumpableAttrs, always_dump="a b"):
        a: str = attr.Factory(str)
        b: str = attr.Factory(str)
        c: str = attr.Factory(str)

    s = yaml.dump(Config())
    assert (
        s
        == """\
!Config
a: ''
b: ''
"""
    )

    class Config(DumpableAttrs, always_dump="*"):
        a: str = attr.Factory(str)
        b: str = attr.Factory(str)

    s = yaml.dump(Config())
    assert (
        s
        == """\
!Config
a: ''
b: ''
"""
    )


# Dataclass load testing


def test_dump_load_aliases():
    """ Ensure dumping and loading `xx=Alias('x')` works.
    Ensure loading `{x=1, xx=1}` raises an error.
    Does not check constructor `Config(xx=1)`."""

    class Config(DumpableAttrs, kw_only=False):
        x: int
        xx = Alias("x")

    # Test dumping
    assert len(attr.fields(Config)) == 1
    cfg = Config(1)
    s = yaml.dump(cfg)
    assert (
        s
        == """\
!Config
x: 1
"""
    )
    assert yaml.load(s) == cfg

    # Test loading
    s = """\
!Config
xx: 1
"""
    assert yaml.load(s) == Config(x=1)

    # Test exception on duplicated parameters.
    s = """\
    !Config
    x: 1
    xx: 1
    """
    with pytest.raises(CorrError):
        yaml.load(s)


def test_dump_load_ignored():
    """ Ensure loading `xx=Ignored` works.
    Does not check constructor `Config(xx=1)`.
    """

    class Config(DumpableAttrs):
        xx = Ignored

    # Test dumping
    assert len(attr.fields(Config)) == 0
    cfg = Config()
    s = yaml.dump(cfg)
    assert (
        s
        == """\
!Config {}
"""
    )
    assert yaml.load(s) == cfg

    # Test loading
    s = """\
!Config
xx: 1
"""
    assert yaml.load(s) == Config()


def test_load_argument_validation():
    """ Ensure that loading config via YAML catches missing and invalid parameters. """

    class Config(DumpableAttrs):
        a: int

    yaml.load(
        """\
!Config
  a: 1
"""
    )

    with pytest.raises(TypeError):
        yaml.load("!Config {}")

    with pytest.raises(TypeError):
        yaml.load(
            """\
!Config
  a: 1
  b: 1
"""
        )


def test_load_post_init():
    """ yaml.load() does not natively call __init__.
    So DumpableAttrs modifies __setstate__ to call __attrs_post_init__. """

    class Foo(DumpableAttrs):
        foo: int

        def __attrs_post_init__(self):
            self.foo = 99

    s = """\
!Foo
foo: 0
"""
    assert yaml.load(s) == Foo(99)


# Test always_dump validation.


def test_always_dump_validate():
    # Validator not implemented.
    # with pytest.raises(AssertionError):
    #     class Foo(DumpableAttrs, always_dump="foo foo"):
    #         foo: int

    with pytest.raises(AssertionError):

        class Foo(DumpableAttrs, always_dump="* foo"):
            foo: int

    with pytest.raises(AssertionError):

        class Foo(DumpableAttrs, always_dump="bar"):
            foo: int


# ruamel.yaml has a unstable and shape-shifting API.
# Test which version numbers have properties we want.


def test_dump_dataclass_order():
    class Config(DumpableAttrs, always_dump="*"):
        a: int = 1
        b: int = 1
        c: int = 1
        d: int = 1
        e: int = 1
        z: int = 1
        y: int = 1
        x: int = 1
        w: int = 1
        v: int = 1

    assert (
        yaml.dump(Config())
        == """\
!Config
a: 1
b: 1
c: 1
d: 1
e: 1
z: 1
y: 1
x: 1
w: 1
v: 1
"""
    )


def test_load_dump_dict_order():
    s = """\
a: 1
b: 1
c: 1
d: 1
e: 1
z: 1
y: 1
x: 1
w: 1
v: 1
"""
    dic = yaml.load(s)
    assert yaml.dump(dic) == s, yaml.dump(dic)


def test_load_list_dict_type():
    """Fails on ruamel.yaml<0.15.70 (CommentedMap/CommentedSeq)."""
    dic = yaml.load("{}")
    assert isinstance(dic, dict)

    lis = yaml.load("[]")
    assert isinstance(lis, list)


def test_list_slice_assign():
    """Crashes on ruamel.yaml<0.15.55 (CommentedSeq)."""
    lis = yaml.load("[]")
    lis[0:0] = list(range(5))
    lis[2:5] = []
