"""Sphinx configuration for the GNSS-R System Simulator documentation.

The documentation tree mirrors the ``SRC/`` source layout (Basilisk-style): every module gets its
own reStructuredText page next to its place in the tree. API signatures are pulled from the module
docstrings via autodoc (NumPy style); the physics / math narrative is hand-written per module.
"""

from __future__ import annotations

import os
import sys

# The simulator is not installed as a package; its entry-point scripts rely on SRC/ being on
# sys.path (this mirrors [tool.pytest.ini_options].pythonpath in pyproject.toml). Do the same here
# so autodoc can import the modules under their bare names (functions.link_budget, ...).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "SRC"))

# Force a headless matplotlib backend so importing modules that pull in matplotlib during the doc
# build does not try to start a Qt event loop (the moving-target entry point does the same at runtime).
os.environ.setdefault("MPLBACKEND", "Agg")

# -- Project information ------------------------------------------------------

project = "GNSS-R System Simulator"
author = "Oliver Hasler, Corrado Chiatante"
copyright = f"2026, {author}"  # noqa: A001
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # pull API docs from the module docstrings
    "sphinx.ext.napoleon",  # NumPy-style docstrings (see [tool.ruff.lint.pydocstyle].convention)
    "sphinx.ext.mathjax",  # LaTeX math rendering for the physics descriptions
    "sphinx.ext.viewcode",  # link API entries to highlighted source
    "sphinx.ext.intersphinx",  # cross-link to the Python / NumPy reference docs
    "myst_parser",  # allow Markdown sources alongside reStructuredText
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Number and display every displayed equation across all module pages, even those without an explicit
# :label:. Keeps the physics/math front-and-centre and referenceable on every page.
math_number_all = True
numfig = True
# Caption labels read "Fig. 6:" / "Table 6:" (colon after the number). numref cross-references keep the
# plain "Table 6" form via an explicit format given at the reference site.
numfig_format = {
    "figure": "Fig. %s:",
    "table": "Table %s:",
}

# -- Autodoc -----------------------------------------------------------------

# Heavy, GUI-bound, or data-downloading leaf dependencies are mocked so the doc build does not need
# the full runtime environment (a display, geospatial data, NASA download clients). Never mock the
# project's own modules -- only third-party leaves.
autodoc_mock_imports = [
    "cartopy",
    "geopandas",
    "pyogrio",
    "PyQt5",
    "podaac_data_subscriber",
    "harmony",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"  # render type hints in the parameter list, not the signature

napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = "GNSS-R System Simulator"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Autodoc tweaks ----------------------------------------------------------

from docutils import nodes  # noqa: E402


def _strip_object_bases(app, doctree):
    """Drop the noise-only "Bases: object" line; keep meaningful bases (Target, ABC, ...)."""
    for para in list(doctree.findall(nodes.paragraph)):
        if para.astext().strip() == "Bases: object":
            para.parent.remove(para)


def setup(app):
    app.connect("doctree-read", _strip_object_bases)
