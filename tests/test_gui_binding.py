import pytest

from corrscope.gui.data_bind import rgetattr, rsetattr, rhasattr


def test_rgetattr():
    """ Test to ensure recursive model access works.
    GUI elements are named "prefix__" "recursive__attr" and bind to recursive.attr.

    https://stackoverflow__com/a/31174427/
    """

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
