"""
Integration tests found in:
- test_cli.py
- test_renderer.py
- test_output.py
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pytest

from corrscope import outputs

if TYPE_CHECKING:
    import pytest_mock


# Pycharm sets cwd to /tests/.
# To ensure tests can find WAV files (based on /), jump from /tests/conftest.py to /.
os.chdir(Path(__file__).parent.parent)


@pytest.fixture
def Popen(mocker: "pytest_mock.MockFixture"):
    """
    - This fixture function is called Popen.
        - pytest names our yield-value Popen too.
    - The yielded Popen imitates `class Popen`, but is actually a MagicMock instance.
        - We can't make Popen a class, because a test uses `Popen.call_args`.
    - To "add class-methods" to Popen, I create a `class MockPopen`
      and give its classmethods to Popen.
    """
    real_Popen = outputs.MyPopen

    class MockPopen(real_Popen):
        exception_on_write: Optional[Exception] = None

        @classmethod
        def set_exception(cls, exc: Exception):
            cls.exception_on_write = exc

        @classmethod
        def __new__(cls, *args, **kwargs):
            popen = mocker.create_autospec(real_Popen)

            popen.stdin = mocker.mock_open()(os.devnull, "wb")
            popen.stdout = mocker.mock_open()(os.devnull, "rb")
            assert popen.stdin != popen.stdout

            if cls.exception_on_write is not None:
                popen.stdin.write.side_effect = cls.exception_on_write

            popen.wait.return_value = 0
            return popen

    # Popen acts exactly the same as MockPopen
    # when calling classmethods or creating instances.
    # But Popen is a MagicMock, and captures Popen() call arguments.

    Popen = mocker.patch.object(outputs, "MyPopen", autospec=True)
    Popen.set_exception = MockPopen.set_exception
    Popen.side_effect = MockPopen.__new__
    yield Popen
