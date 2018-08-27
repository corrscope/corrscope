import os
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import pytest_mock


@pytest.fixture
def Popen(mocker: 'pytest_mock.MockFixture'):
    Popen = mocker.patch.object(subprocess, 'Popen', autospec=True)
    popen = Popen.return_value

    popen.stdin = open(os.devnull, "wb")
    popen.stdout = open(os.devnull, "rb")
    popen.wait.return_value = 0

    yield Popen
