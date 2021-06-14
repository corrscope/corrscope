""" poetry run func_name """
import shlex
import webbrowser

# Obtain path from package.dist-info/entry_points.txt
def run(path, arg_str):
    module, func = path.split(":")
    argv = shlex.split(arg_str)
    exec(
        f"""\
from {module} import {func}
{func}({argv})
"""
    )


# public:
def dcover():
    """Run coverage and diff-cover."""
    cover()
    diff()


def rcover():
    """Run coverage and report."""
    cover()
    report()


def hcover():
    """Run coverage and open HTML."""
    cover()
    html()


# public helpers:
def cover():
    run("pytest:main", "--tb=short --cov=corrscope")


def diff():
    # argv[0:]
    run("diff_cover.tool:main", "diff-cover coverage.xml")


def report():
    run("coverage.cmdline:main", "report")


def html():
    run("coverage.cmdline:main", "html")
    webbrowser.open("htmlcov/index.html")


"""
export MONKEYTYPE_TRACE_MODULES=corrscope
monkeytype run `which pytest`
// monkeytype run -m corrscope
monkeytype list-modules | xargs -I % -n 1 sh -c 'monkeytype apply % 2>&1 | tail -n4'
"""
