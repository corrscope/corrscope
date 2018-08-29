import os
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import pytest_mock


@pytest.fixture
def Popen(mocker: 'pytest_mock.MockFixture'):
    real_Popen = subprocess.Popen

    def popen_factory(*args, **kwargs):
        popen = mocker.create_autospec(real_Popen)
        popen.stdin = open(os.devnull, "wb")
        popen.stdout = open(os.devnull, "rb")
        popen.wait.return_value = 0
        return popen

    Popen = mocker.patch.object(subprocess, 'Popen', autospec=True)
    Popen.side_effect = popen_factory
    yield Popen
