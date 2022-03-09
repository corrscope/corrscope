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


"""
export MONKEYTYPE_TRACE_MODULES=corrscope
monkeytype run `which pytest`
// monkeytype run -m corrscope
monkeytype list-modules | xargs -I % -n 1 sh -c 'monkeytype apply % 2>&1 | tail -n4'
"""
