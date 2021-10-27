"""Color definitions & functionality for matplotlib and the terminal."""

import builtins
import contextlib

import colorama
import matplotlib as mpl
import numpy as np

# Makes stdout/err color codes work on windows too.
colorama.init()


#########################################
# Colouring for the terminal / console
#########################################
def color_text(text, *color_codes):
    """Color a string for the terminal.

    Multiple `color_codes` can be combined.
    Look them up in `colorama.Back` and `colorama.Fore`.
    Use the single code `None` for no coloring.
    """
    c0 = colorama.Style.RESET_ALL

    if (not color_codes) or color_codes == ("default",):
        cc = [colorama.Style.BRIGHT, colorama.Fore.BLUE]
    else:
        # Foreground colors can be specified as strings
        cc = [getattr(colorama.Fore, c.upper(), c) for c in color_codes if c]

    if not cc:
        return text
    else:
        return "".join(cc) + text + c0


@contextlib.contextmanager
def coloring(*color_codes):
    """Color printing using 'with'.

    Example:
        with coloring(colorama.Fore.GREEN):
           print("--- This is in color ---")
    """
    orig_print = builtins.print

    def _print(*args, sep=" ", end="\n", flush=True, **kwargs):
        # Implemented with a single print statement, so as to
        # puts the trailing terminal code before newline.
        text = sep.join([str(k) for k in args])
        text = color_text(text, *color_codes)
        orig_print(text, end=end, flush=flush, **kwargs)

    try:
        builtins.print = _print
        yield
    finally:
        builtins.print = orig_print


def stripe(rows, inds=...):
    if not isinstance(rows, list):
        rows = rows.splitlines()
    inds = np.arange(len(rows))[inds]
    for i, row in enumerate(rows):
        if i in inds and i % 2:
            rows[i] = color_text(row, colorama.Fore.BLACK, colorama.Back.WHITE)
    rows = "\n".join(rows)
    return rows


#########################################
# Colouring for matplotlib
#########################################
# Matlab (new) colors.
ml_colors = np.array(
    [[0.   , 0.447, 0.741],
     [0.85 , 0.325, 0.098],
     [0.929, 0.694, 0.125],
     [0.494, 0.184, 0.556],
     [0.466, 0.674, 0.188],
     [0.301, 0.745, 0.933],
     [0.635, 0.078, 0.184]])

# Load into matplotlib color dictionary
for code, color in zip('boyvgcr', ml_colors):
    mpl.colors.ColorConverter.colors['ml'+code] = color
    mpl.colors.colorConverter. cache['ml'+code] = color

# Seaborn colors
sns_colors = np.array(
    [[0.298, 0.447, 0.69 ],
     [0.333, 0.658, 0.407],
     [0.768, 0.305, 0.321],
     [0.505, 0.447, 0.698],
     [0.8  , 0.725, 0.454],
     [0.392, 0.709, 0.803],
     [0.1  , 0.1  , 0.1  ],
     [1.   , 1.   , 1.   ]])
# Overwrite default color codes
for code, color in zip('bgrmyckw', sns_colors):
    mpl.colors.colorConverter.colors[code] = color
    mpl.colors.colorConverter. cache[code] = color


# MPL colors -- cheat sheet:
# --------------
# Equivalent:
# cmap = plt.get_cmap("tab10")
# cmap = plt.cm.get_cmap("tab10")
# cmap = mpl.cm.get_cmap("tab10")
# cmap = mpl.cm.tab10
# Equivalent:
# c = cmap( cmap.N//2 ) # using int
# c = cmap( .5 )        # using float
# Equivalent:
# clist = cmap.colors # (only for ListedColormap) => 2D tuple,
# clist = cmap(np.arange(0,.9,.1))              # => 2D ndarray
#
# Choosing cmap: https://matplotlib.org/3.3.1/tutorials/colors/colormaps.html
# Color codes: https://matplotlib.org/tutorials/colors/colors.html
