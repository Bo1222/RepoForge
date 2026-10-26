import pytest

from repoforge.actions.test_targets import is_verifiable_test_module
from repoforge.repository.repository import InMemRepository


@pytest.fixture
def repository():
    repo = InMemRepository()
    repo.save_file("tests/test_ok.py", "def test_ok():\n    assert True\n")
    repo.save_file("tests/test_async.py", "async def test_async():\n    assert True\n")
    repo.save_file("tests/test_case.py", "import unittest\n\nclass TestThing(unittest.TestCase):\n    def test_x(self):\n        pass\n")
    repo.save_file("tests/test_class.py", "class TestThing:\n    def test_x(self):\n        pass\n")
    repo.save_file("package/__init__.py", "def test_not_really():\n    pass\n")
    repo.save_file("django/test/signals.py", "def receiver():\n    pass\n")
    repo.save_file("tests/custom_pk/fields.py", "class Field:\n    pass\n")
    repo.save_file("tests/roots/conf.py", "def test_fixture():\n    pass\n")
    repo.save_file("tests/test_broken.py", "def test_broken(:\n    pass\n")
    return repo


@pytest.mark.parametrize("path", [
    "tests/test_ok.py",
    "tests/test_async.py",
    "tests/test_case.py",
    "tests/test_class.py",
])
def test_accepts_real_test_modules(repository, path):
    assert is_verifiable_test_module(repository, path)


@pytest.mark.parametrize("path", [
    "package/__init__.py",
    "django/test/signals.py",
    "tests/custom_pk/fields.py",
    "tests/roots/conf.py",
    "tests/test_broken.py",
])
def test_rejects_support_or_product_modules(repository, path):
    assert not is_verifiable_test_module(repository, path)


def test_discover_ranks_related_test_module(repository):
    from repoforge.actions.test_targets import discover_test_targets
    repository.save_file("tests/test_numbers.py", "from package.numbers import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    repository.save_file("tests/test_strings.py", "def test_string():\n    assert True\n")
    ranked = discover_test_targets(repository, ["package/numbers.py"],
                                   ["tests/test_numbers.py", "tests/test_strings.py"])
    assert ranked[0] == "tests/test_numbers.py"


def test_require_relation_drops_unrelated_test_modules(repository):
    from repoforge.actions.test_targets import discover_test_targets
    repository.save_file("tests/test_numbers.py", "from package.numbers import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    ranked = discover_test_targets(
        repository, ["package/numbers.py"],
        ["tests/test_numbers.py", "tests/test_ok.py", "tests/test_async.py"],
        require_relation=True)
    assert ranked and ranked[0] == "tests/test_numbers.py"
    assert "tests/test_ok.py" not in ranked
    assert "tests/test_async.py" not in ranked


def test_is_related_test_uses_name_or_import(repository):
    from repoforge.actions.test_targets import is_related_test
    repository.save_file("package/numbers.py", "def add(a, b):\n    return a + b\n")
    repository.save_file("tests/test_numbers.py", "from package.numbers import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    assert is_related_test(repository, "tests/test_numbers.py", ["package/numbers.py"])
    assert not is_related_test(repository, "tests/test_ok.py", ["package/numbers.py"])
