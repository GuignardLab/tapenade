from organoid import Organoid


def test_organoid_creation():
    """You should write tests here!"""
    test_obj = Organoid(1)
    assert test_obj.arg == 1


def test_organoid_manipulation():
    """You should write tests here!"""
    test_obj = Organoid(1)
    test_obj.arg += 2
    assert test_obj.arg == 3
