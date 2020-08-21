"""Provide (terminal and matplotlib) color definitions and functionality."""

from dapper import *

#########################################
# Colouring for the terminal / console
#########################################
import colorama
colorama.init() # Makes stdout/err color codes work on windows too.
from colorama import Fore as cFG # Foreground color codes
from colorama import Back as cBG # Background color codes
import contextlib

def color_text(text, *color_codes):

    if len(color_codes)==0:
        color_codes = [colorama.Style.BRIGHT, cFG.BLUE]

    return "".join(color_codes) + text + colorama.Style.RESET_ALL


@contextlib.contextmanager
def coloring(*color_codes):
    """Color printing using 'with'.
    
    Example:
    >>> with coloring(cFG.GREEN): print("This is in color")
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


#########################################
# Colouring for matplotlib
#########################################
# Matlab (new) colors.
ml_colors = np.array(np.matrix("""
     0    0.4470    0.7410;
0.8500    0.3250    0.0980;
0.9290    0.6940    0.1250;
0.4940    0.1840    0.5560;
0.4660    0.6740    0.1880;
0.3010    0.7450    0.9330;
0.6350    0.0780    0.1840 
"""))
# Load into matplotlib color dictionary
for code, color in zip('boyvgcr', ml_colors):
    mpl.colors.ColorConverter.colors['ml'+code] = color
    mpl.colors.colorConverter.cache ['ml'+code] = color

# Seaborn colors
sns_colors = np.array(np.matrix("""
0.298 , 0.447 , 0.690 ; 
0.333 , 0.658 , 0.407 ; 
0.768 , 0.305 , 0.321 ; 
0.505 , 0.447 , 0.698 ; 
0.8   , 0.725 , 0.454 ; 
0.392 , 0.709 , 0.803 ; 
0.1   , 0.1   , 0.1   ; 
1.0   , 1.0   , 1.0    
"""))
# Overwrite default color codes
for code, color in zip('bgrmyckw', sns_colors):
    mpl.colors.colorConverter.colors[code] = color
    mpl.colors.colorConverter.cache [code] = color
