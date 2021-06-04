"""Dirty magic. Should never be used. Ever."""

# Answers SO questions # 54476645, 28244921

import ast
import inspect

from dapper.tools.colors import coloring


def get_call():
    """Get calling statement (even if it is multi-lined), 2 frames back.

    NB: returns full lines (may include junk before/after calls)
        coz real parsing (brackets, commas, backslash, etc) is complicated.

    Also returns caller args' names, and caller's namespace.

    Old method was based on looping backwards until func_name is found.
    But that doesn't work since f_lineno now (since python 3.8) yields
    the 1st line of a multi-line call (https://bugs.python.org/issue38283).
    New method (using AST) is actually a good deal more robust, because
    we get actual argnames and dont need to resort to regex.
    """
    # Introspection
    f0         = inspect.currentframe()  # this frame
    f1         = f0.f_back  # frame of the function whose call we want to get
    f2         = f1.f_back  # frame of the calling
    code, shift = inspect.getsourcelines(f2)
    func_name  = f1.f_code.co_name

    # Using stack instead
    # iFrame = 2
    # frame,filename,n1,fname,lines,index = inspect.stack()[iFrame]

    # Note: it may be that using one of
    #     code = inspect.getsource(f2)
    #     code = open(f2.f_code.co_filename).read().splitlines(True)
    # is less problematic with regards to
    # - indentation, which gets parsed by ast.NodeVisitor
    # - fix01 and fix10
    # because it reads the entire module or something.
    # On the other hand, it might be less efficient,
    # and less general (does it work when the call is in the interpreter?)
    # Need a large battery of tests to really decide what's best.

    # Get call's line number
    n1  = f2.f_lineno
    n1 -= shift
    # I really don't know why these become necessary
    fix01 = 0 if shift else 1
    fix10 = 1 if shift else 0
    # print("n1:",n1)
    # print("code[n1-fix01]:\n",code[n1])

    # Walk syntax tree
    class Visitor(ast.NodeVisitor):
        """Get info on call if name and lineno match."""

        # Inspiration for relevant parts of AST:
        # https://docs.python.org/3/library/ast.html#abstract-grammar
        # https://docs.python.org/3/library/ast.html#ast.Call
        # http://alexleone.blogspot.com/2010/01/python-ast-pretty-printer.html
        def visit_Call(self, node):
            node_id = getattr(node.func, "id", None)
            if node_id == func_name:
                if node.lineno == (n1 + fix10):
                    assert info == {}, "Matched with multiple calls."
                    info["n1"] = node.lineno
                    info["c1"] = node.col_offset
                    info["n2"] = node.end_lineno
                    info["c2"] = node.end_col_offset
                    try:
                        info["argnames"] = [arg.id for arg in node.args]
                    except AttributeError:
                        pass  # argnames will not be present
            self.generic_visit(node)

    info = {}
    Visitor().visit(ast.parse("".join(code)))
    assert "n2" in info, "Failed to find caller in its file."

    call_text = "".join(code[n1-fix01: info["n2"]])
    call_text = call_text.rstrip()  # rm trailing newline

    return call_text, info.get("argnames", None), f2.f_locals


# TODO 4: fails on python 3.7 and older.
# I believe there is a version in the git history that works with py <= 3.7.
# But maybe we should not use magic any more?
def magic_naming(*args, **kwargs):
    """Convert args (by their names in the call) to kwargs.

    Example:
    >>> a, b = 1, 2
    >>> magic_naming(a, b, c=3)
    {'a': 1, 'b': 2, 'c': 3}
    """
    call, argnames, locvars = get_call()
    assert len(args) == len(argnames), "Something's gone wrong."

    # Obsolete (works with call rather than argnames).
    # Insert args. Matches arg to a name by
    # # - M1: id to a variable in the local namespace, and
    # # - M2: the presence of said variable in the call.
    # for i,arg in enumerate(args):
    #     # List of candidate names for arg i
    #     nn = [name for name in locvars if locvars[name] is arg]         # (M1)
    #     nn = [name for name in nn if re.search(r"\b"+name+r"\b", call)] # (M2)
    #     if not nn:
    #         raise RuntimeError("Couldn't find the name for "+str(arg))
    #     # Associating arg to ALL matching names.
    #     for name in nn:
    #         dct[name] = arg

    dct = {name: arg for arg, name in zip(args, argnames)}
    dct.update(kwargs)
    return dct


def spell_out(*args):
    """Print (args) including variable names.

    Example
    -------
    >>> spell_out(3*2)
    3*2:
    6
    """
    call, _, loc = get_call()

    # Find opening/closing brackets
    left  = call. find("(")
    right = call.rfind(")")

    # Print header
    import sys
    c = None if "pytest" in sys.modules else "blue"
    with coloring(c):
        print(call[left+1:right] + ":")

    # Print (normal)
    print(*args)


if __name__ == "__main__":
    lst = [chr(97+i) for i in range(7)]
    dct2 = {c: c for c in lst}
    a, b, c, d, e, f, g = lst

    print(magic_naming(a, b,
                       c, d,  # fawef
                       e, f, g))

    spell_out(a, b*2, 3*4)

    ###########
    #  tests  #
    ###########
    # pytest reports failure on the following assertions.
    # But intorspection is brittle, so that's not surprising.

    d2 = magic_naming(a, b, c, d, e, f, g)
    assert d2 == dct2

    # Ugly call
    assert \
        {"b": "b", "a": 3} == \
        magic_naming(b, a=3,
                     )
