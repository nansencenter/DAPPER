"""
# Development

``
# Profiling.
# Launch python script: $ kernprof -l -v myprog.py
# Functions decorated with 'profile' from below will be timed.
try:
    import builtins
    profile = builtins.profile     # will exists if launched via kernprof
except AttributeError:
    def profile(func): return func # provide a pass-through version.
``
"""
