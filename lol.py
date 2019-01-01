import matplotlib as mpl
import os

'''\
font_manager.py: fonts are stored in
- get_cachedir() == _get_config_or_cache_dir(_get_xdg_cache_dir())

_get_xdg_cache_dir uses first of
- $MPLCONFIGDIR
- _get_xdg_cache_dir()

uses
- get_home()

_get_config_or_cache_dir defaults to
- _create_tmp_config_dir 
'''

evals = """
mpl.get_cachedir()

os.environ.get('MPLCONFIGDIR')
mpl._get_config_or_cache_dir(mpl._get_xdg_cache_dir())

mpl._get_xdg_cache_dir()

mpl.get_home()
""".split()


for s in evals:
    print(f'{s} -> {eval(s)}')
