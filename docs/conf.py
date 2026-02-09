# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Symmetric Learning"
copyright = "2025, Daniel Felipe Ordoñez Apraez"
author = "Daniel Felipe Ordoñez Apraez"
from importlib.metadata import version as get_version

release = get_version("symm-learning")
version = ".".join(release.split(".")[:3])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
docs_branch = os.environ.get("DOCS_BRANCH", os.environ.get("GITHUB_REF_NAME", "local"))
docs_banner = f"v{version} - {docs_branch}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    # 'numpydoc',
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = f"{project} v{version} Docs [{docs_branch}]"

add_module_names = False


html_theme_options = {
    "announcement": f"<strong>{docs_banner}</strong>",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Danfoa/symmetric_learning",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_align": "content",
    "collapse_navigation": True,
    "show_nav_level": 1,
    "navigation_depth": 4,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
}

html_css_files = ["custom.css"]


# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "escnn": ("https://quva-lab.github.io/escnn/", None),
}

autoclass_content = "both"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
}
