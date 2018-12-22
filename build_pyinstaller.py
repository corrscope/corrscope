import os
import subprocess
import sys

# To write graph, pass --log-level=DEBUG

# pyinstaller corrscope/__main__.py --name corrscope -y
args = 'pyinstaller corrscope.spec -y'.split() + sys.argv[1:]

subprocess.run(args)
