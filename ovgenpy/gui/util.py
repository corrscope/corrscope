import matplotlib.colors


def color2hex(color):
    return matplotlib.colors.to_hex(color, keep_alpha=False)
