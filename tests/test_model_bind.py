import pytest
from pytest_cases import pytest_fixture_plus

from corrscope.gui.model_bind import rgetattr, rsetattr, rhasattr, flatten_attr


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


@pytest_fixture_plus
@pytest.mark.parametrize("s", ["__", "."])
def separator(s: str) -> str:
    return s


def test_rgetattr(separator):
    """Test to ensure recursive model access works.
    GUI elements are named f"prefix{_}" "recursive{_}attr" and bind to recursive.attr.

    https://stackoverflow{_}com/a/31174427/
    """
    p = Person()
    _ = separator

    # Test rgetattr(present)
    assert rgetattr(p, f"pet{_}species") == "Dog"
    assert rgetattr(p, f"pet{_}species", object()) == "Dog"

    # Test rgetattr(missing)
    assert rgetattr(p, f"pet{_}ghost{_}species", "calico") == "calico"
    with pytest.raises(AttributeError):
        # Without a default argument, `rgetattr`, like `getattr`, raises
        # AttributeError when the dotted attribute is missing
        print(rgetattr(p, f"pet{_}ghost{_}species"))

    # Test rsetattr()
    rsetattr(p, f"pet{_}name", "Sparky")
    rsetattr(p, f"residence{_}type", "Apartment")
    assert p.pet.name == "Sparky"
    assert p.residence.type == "Apartment"

    # Test rhasattr()
    assert rhasattr(p, f"pet")
    assert rhasattr(p, f"pet{_}name")

    # Test rhasattr(levels of missing)
    assert not rhasattr(p, f"pet{_}ghost")
    assert not rhasattr(p, f"pet{_}ghost{_}species")
    assert not rhasattr(p, f"ghost")
    assert not rhasattr(p, f"ghost{_}species")


def test_flatten_attr(separator):
    p = Person()
    _ = separator

    # Test nested
    flat, name = flatten_attr(p, f"pet{_}name")
    assert flat is p.pet
    assert name == "name"

    # Test 1 level
    flat, name = flatten_attr(p, f"pet")
    assert flat is p
    assert name == "pet"


def test_rgetattr_broken(separator):
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
    _ = separator

    result = rgetattr(object(), f"nothing{_}imag", 1)
    assert result == 1, result
