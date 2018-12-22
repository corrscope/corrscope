import glob

from PyInstaller.building.build_main import Analysis
from PyInstaller.building.api import PYZ, EXE, COLLECT

from PyInstaller.building.datastruct import TOC

block_cipher = None


dir = 'corrscope/gui'
includes = glob.glob('corrscope/gui/*.ui')
assert includes

datas = [(include, dir) for include in includes]

a = Analysis(['corrscope\\__main__.py'],
             pathex=['.'],
             binaries=[],
             datas=datas,
             hiddenimports=['corrscope.gui.__init__'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[
                 'FixTk',
                 'tcl',
                 'tk',
                 '_tkinter',
                 'tkinter',
                 'Tkinter'
             ],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)


# Some dirs are included by PyInstaller hooks and must be removed after the fact.
path_excludes = (
    # Matplotlib
    '''
    mpl-data/fonts
    mpl-data/images
    mpl-data/sample_data
    '''
    # Scipy (FIXME not excluded and breaks anyway)
    '''
    scipy/sparse
    scipy/linalg
    scipy/spatial
    scipy/special
    scipy/interpolate
    scipy/optimize
    scipy/ndimage
    '''
    # PyQt
    '''
    Qt5DBus.dll
    Qt5Network.dll
    Qt5Quick.dll
    Qt5Qml.dll
    Qt5Svg.dll
    Qt5WebSockets.dll
    '''
    # QML file list taken from https://github.com/pyinstaller/pyinstaller/blob/0f31b35fe96de59e1a6faf692340a9ef93492472/PyInstaller/hooks/hook-PyQt5.py#L55
    '''
    libEGL.dll libGLESv2.dll d3dcompiler_ opengl32sw.dll
    '''
).split()
path_excludes = {s.lower() for s in path_excludes}

def path_contains(path: str) -> bool:
    path = path.replace('\\', '/').lower()
    ret = any(x in path for x in path_excludes)
    return ret


# A TOC appears to be a list of tuples of the form (name, path, typecode).
# In fact, it's an ordered set, not a list.
# A TOC contains no duplicates, where uniqueness is based on name only.
def strip(arr: TOC):
    arr[:] = [
        (name, path, typecode) for (name, path, typecode) in arr
        if not path_contains(path)
    ]


strip(a.datas)
strip(a.binaries)


pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='corrscope',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='corrscope')
