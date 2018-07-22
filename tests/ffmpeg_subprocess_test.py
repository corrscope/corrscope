import glob
import shlex
import subprocess

templates = ['ffplay -autoexit -f image2pipe -framerate 60 -i -']
args = [arg
        for template in templates
        for arg in shlex.split(template)]

popen = subprocess.Popen(args, stdin=subprocess.PIPE)
stream = popen.stdin

imgs = glob.glob('images/*.png')
imgs.sort()
for img in imgs * 100:
    with open(img, 'rb') as f:
        stream.write(f.read())    # FIXME https://docs.python.org/3/library/subprocess.html#subprocess.Popen.stdin

# Warning: Use communicate() rather than .stdin.write, .stdout.read or .stderr.read
# to avoid deadlocks due to any of the other OS pipe buffers filling up and blocking
# the child process.

# communicate(): Interact with process: Send data to stdin. Read data from stdout and
# stderr, until end-of-file is reached.

# nope instant reject

# https://stackoverflow.com/questions/9886654/difference-between-communicate-and-stdin-write-stdout-read-or-stderr-read

stream.close()
popen.wait()
