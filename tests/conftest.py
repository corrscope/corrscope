"""
Integration tests found in:
- test_cli.py
- test_renderer.py
- test_output.py
"""

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pytest

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
    real_Popen = subprocess.Popen
    exception_on_write: Optional[Exception] = None

    def set_exception(exc: Exception):
        nonlocal exception_on_write
        exception_on_write = exc

    def new_popen(*args, **kwargs):
        popen = mocker.create_autospec(real_Popen)

        popen.stdin = mocker.mock_open()(os.devnull, "wb")
        popen.stdout = mocker.mock_open()(os.devnull, "rb")
        assert popen.stdin != popen.stdout

        if exception_on_write is not None:
            popen.stdin.write.side_effect = exception_on_write

        popen.wait.return_value = 0
        return popen

    # Popen acts exactly the same as MockPopen
    # when calling classmethods or creating instances.
    # But Popen is a MagicMock, and captures Popen() call arguments.

    Popen = mocker.patch.object(subprocess, "Popen", autospec=True)
    Popen.set_exception = set_exception
    Popen.side_effect = new_popen

    # ffprobe_is_mono() calls subprocess.run().
    run = mocker.patch.object(subprocess, "run", autospec=True)

    def do_run(args, *pos, **kwargs):
        return subprocess.CompletedProcess(args, 0, b"1\n", b"")

    run.side_effect = do_run

    yield Popen
