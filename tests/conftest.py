"""
Integration tests found in:
- test_cli.py
- test_output.py
"""

import os
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import pytest_mock


# TODO patch outputs.Popen to avoid breaking multiprocessing
@pytest.fixture
def Popen(mocker: "pytest_mock.MockFixture"):
    real_Popen = subprocess.Popen

    def popen_factory(*args, **kwargs):
        popen = mocker.create_autospec(real_Popen)

        popen.stdin = mocker.mock_open()(os.devnull, "wb")
        popen.stdout = mocker.mock_open()(os.devnull, "rb")
        assert popen.stdin != popen.stdout

        popen.wait.return_value = 0
        return popen

    Popen = mocker.patch.object(subprocess, "Popen", autospec=True)
    Popen.side_effect = popen_factory
    yield Popen
