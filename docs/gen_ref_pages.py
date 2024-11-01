"""Generate the code reference pages."""
# Based on https://mkdocstrings.github.io/recipes/

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "dapper"

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1] or src.parts[-1:]
        if not parts:
            # we're in root pkg
            parts = src.parts[-1:]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    # PS: rm mkdocs_gen_files to get to inspect actual .md files
    # NB: will (over)write in docs/ folder.
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # Explicitly set the title to avoid mkdocs capitalizing
        # names and removing underscores (only applies to files)
        print(f"# {parts[-1]}", file=fd)

        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, ".." / path.relative_to(root))

# > So basically, you can use the literate-nav plugin just for its ability to
# > infer only sub-directories, without ever writing any actual "literate navs".
# with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
#     nav_file.writelines(nav.build_literate_nav())
