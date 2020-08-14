"""Load rc: default settings"""

class JsonDict(dict):
    """Provide json pretty-printing"""
    np_short = True

    def __repr__(self):
        lines = json.dumps(self, indent=4, sort_keys=False, default=self.fallback)
        lines = lines.split("\n")
        lines = self.de_escape_newlines(lines)
        cropr = lambda t: t[:80] + ("" if len(t)<80 else "...")
        lines = [cropr(ln) for ln in lines]
        return "\n".join(lines)

    def __str__(self):
        return repr(self)

    def fallback(self, obj):
        if JsonDict.np_short and 'numpy.ndarray' in str(type(obj)):
            return f"ndarray, shape {obj.shape}, dtype {obj.dtype}"
        else:
            return str(obj)

    def de_escape_newlines(self,lines):
        """De-escape newlines. Include current indent."""
        new = []
        for line in lines:
            if "\\n" in line:
                hang = 2
                ____ = " " * (len(line) - len(line.lstrip()) + 2)
                line = line.replace('": "', '":\n'+____) # add newline at :
                line = line.replace('\\n', '\n'+____) # de-escape newlines
                line = line.split("\n")
                line[-1] = line[-1].rstrip('"')
            else:
                line = [line]
            new += line
        return new

class DotDict(JsonDict):
    """Dict that *also* supports attribute (dot) access.

    Benefit compared to a dict:

     - Verbosity of ``d['a']`` vs. ``d.a``.
     - Includes ``JsonDict``.

    DotDict is not very hackey, and is quite robust.
    Similar constructs are quite common, eg IPython/utils/ipstruct.py.

    Main inspiration: stackoverflow.com/a/14620633
    """
    def __init__(self,*args,**kwargs):
        "Init like a normal dict."
        super(DotDict, self).__init__(*args,**kwargs) # Make a (normal) dict
        self.__dict__ = self                          # Assign it to self.__dict__


