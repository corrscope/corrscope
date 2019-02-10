import pytest

from corrscope.gui.data_bind import rgetattr, rsetattr, rhasattr, flatten_attr


class Person(object):
    def __init__(self):
        self.pet = Pet()
        self.residence = Residence()


class Pet(object):
    def __init__(self, name="Fido", species="Dog"):
        self.name = name
        self.species = species


class Residence(object):
    def __init__(self, type="House", sqft=None):
        self.type = type
        self.sqft = sqft


def test_rgetattr():
    """ Test to ensure recursive model access works.
    GUI elements are named "prefix__" "recursive__attr" and bind to recursive.attr.

    https://stackoverflow__com/a/31174427/
    """
    p = Person()

    # Test rgetattr(present)
    assert rgetattr(p, "pet__species") == "Dog"
    assert rgetattr(p, "pet__species", object()) == "Dog"

    # Test rgetattr(missing)
    assert rgetattr(p, "pet__ghost__species", "calico") == "calico"
    with pytest.raises(AttributeError):
        # Without a default argument, `rgetattr`, like `getattr`, raises
        # AttributeError when the dotted attribute is missing
        print(rgetattr(p, "pet__ghost__species"))

    # Test rsetattr()
    rsetattr(p, "pet__name", "Sparky")
    rsetattr(p, "residence__type", "Apartment")
    assert p.pet.name == "Sparky"
    assert p.residence.type == "Apartment"

    # Test rhasattr()
    assert rhasattr(p, "pet")
    assert rhasattr(p, "pet__name")

    # Test rhasattr(levels of missing)
    assert not rhasattr(p, "pet__ghost")
    assert not rhasattr(p, "pet__ghost__species")
    assert not rhasattr(p, "ghost")
    assert not rhasattr(p, "ghost__species")


def test_flatten_attr():
    p = Person()

    # Test nested
    flat, name = flatten_attr(p, "pet__name")
    assert flat is p.pet
    assert name == "name"

    # Test 1 level
    flat, name = flatten_attr(p, "pet")
    assert flat is p
    assert name == "pet"


def test_rgetattr_broken():
    """
    rgetattr(default) fails to short-circuit/return on the first missing attribute.
    I never use rgetattr(default) so I won't bother fixing the bug.

    Wrong answer:
    - None.foo AKA 1
    - 1.bar AKA 1
    - 1.imag == 0

    Right answer:
    - None.foo AKA return 1 to caller
    """

    result = rgetattr(object(), "nothing__imag", 1)
    assert result == 1, result
