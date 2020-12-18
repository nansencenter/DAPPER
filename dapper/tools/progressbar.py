"""Make `progbar` (convenience wrapper to `tqdm`) and `read1`."""

import inspect
import os
import sys

import tqdm

disable_progbar = False


def pdesc(desc):
    "Get progbar description by introspection."

    if desc is not None:
        return desc

    stack = inspect.stack()
    # Gives FULL REPR:
    # stack[4].frame.f_locals['name_hook']

    # Go look above in the stack for a name_hook.
    for level in range(2, 6):
        try:
            locals = stack[level].frame.f_locals
        except IndexError:
            pass
        else:
            if 'pb_name_hook' in locals:
                name = locals['pb_name_hook']
                break
    else:
        # Otherwise: just get name of what's
        # calling progbar (i.e. stack[2])
        name = stack[2].function

    return name


def progbar(iterable, desc=None, leave=1, **kwargs):
    "Prints a nice progress bar in the terminal"
    if disable_progbar:
        return iterable
    else:
        desc = pdesc(desc)
        # if is_notebook:
        # This fails in QtConsole (which also yields is_notebook==True)
        # return tqdm.tqdm_notebook(iterable,desc=desc,leave=leave)
        # else:
        return tqdm.tqdm(iterable, desc=desc, leave=leave,
                         smoothing=0.3, dynamic_ncols=True, **kwargs)
        # Printing during the progbar loop (may occur with error printing)
        # can cause tqdm to freeze the entire execution.
        # Seemingly, this is caused by their multiprocessing-safe stuff.
        # Disable this, as per github.com/tqdm/tqdm/issues/461#issuecomment-334343230
        # pb = tqdm.tqdm(...)
        # try: pb.get_lock().locks = []
        # except AttributeError: pass
        # return pb


#########################################
# Make read1()
#########################################
# Non-blocking, non-echo read1 from stdin.

# Set to True when stdin or term settings isn't supported, for example when:
#  - running via py.test
#  - multiprocessing
# Btw, multiprocessing also doesn't like tqdm itself.
disable_user_interaction = False

try:
    # Linux. See Misc/read1_trials.py
    import termios

    def set_term_settings(TS):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, TS)

    def new_term_settings():
        """Make stdin.read non-echo and non-block"""
        # NB: This makes ipython quit (by pressing C^D twice) hang!
        #     So it should be undone before returning to prompt.
        #     Thus, setting/restoring at import/atexit is not viable.

        TS_old = termios.tcgetattr(sys.stdin)
        TS_new = termios.tcgetattr(sys.stdin)

        # Make tty non-echo.
        TS_new[3] = TS_new[3] & ~(termios.ECHO | termios.ICANON)

        # Make tty non-blocking.
        TS_new[6][termios.VMIN] = 0
        TS_new[6][termios.VTIME] = 0

        set_term_settings(TS_new)
        return TS_old

    try:
        # Test set/restore settings
        TS_old = new_term_settings()
        set_term_settings(TS_old)

        orig_progbar = progbar

        # Wrap the progressbar generator so as to temporarily set term settings.
        # Alternative solution: set/restore term settings in assimilate()
        # of the da_method decorator. But that gets bloated, and anyways the logic
        # of user-control of liveplots kindof belongs together with a progressbar.
        def progbar(iterable, desc=None, leave=1, **kwargs):
            if not disable_user_interaction:
                TS_old = new_term_settings()
            try:
                for i in orig_progbar(iterable, pdesc(desc), leave, **kwargs):
                    yield i
            except GeneratorExit:
                # Allows code below to run even if caller raised exception
                # NB: Fails if caller is in a list-comprehesion! Why?
                pass
            # Restore both for normal termination or exception (propagates).
            if not disable_user_interaction:
                set_term_settings(TS_old)

        def _read1():
            return os.read(sys.stdin.fileno(), 1)

    except:  # noqa
        # Fails in non-terminal environments
        disable_user_interaction = True

except ImportError:
    # Windows
    try:
        import msvcrt  # noqa

        def _read1():
            if msvcrt.kbhit():
                return msvcrt.getch()
            else:
                return None

    except ImportError:
        disable_user_interaction = True


def read1():
    "Get 1 character. Non-blocking, non-echoing."
    if disable_user_interaction:
        return None
    return _read1()
