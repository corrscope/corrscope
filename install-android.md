# Android installation

Installing corrscope on Android is experimental and not fully supported. You can report issues on GitHub or Discord, and I will attempt to resolve them if possible.

First install termux from https://f-droid.org/en/packages/com.termux/.

## Installing command-line corrscope

To install, open Termux and run:

```sh
pkg upgrade  # apt update+upgrade
apt install ffmpeg matplotlib python-numpy
pip install corrscope
# omit corrscope[qt5] since it requires an incompatible qt5 packaging. qtpy can still find the system pyqt5.
# when installing with pipx, use --system-site-packages.
```

To run, open Termux and run:

```sh
corr --help
# ...
corr file.yaml --render file.mp4
```

## Installing GUI corrscope with [termux-x11](https://github.com/termux/termux-x11)

To install, open Termux and run:

```sh
pkg upgrade && apt install x11-repo  # apt update+upgrade
apt install termux-x11-nightly pyqt5 ffplay xfwm4
```
- Download and install https://github.com/termux/termux-x11/releases/download/nightly/app-arm64-v8a-debug.apk

To run, open Termux and run:

```sh
termux-x11 :1 -xstartup xfwm4 -dpi 120 & DISPLAY=:1 corr; kill %1
```
- To control the size of items on screen, change `-dpi 120` to another value. (The default DPI is 96; change this value in multiples of 24.)
- Then return to the Android home screen and open the "Termux:X11" app. This should show the corrscope UI.

When terminating a preview, you may get a popup error because ffmpeg/play on Android fails to terminate properly. The dialog can be ignored, and there is no solution at this time.
