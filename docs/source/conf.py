# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from importlib.metadata import PackageNotFoundError, version as package_version

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

project = 'BAccMod'
copyright = '2025, Mathieu de Bony de Lavergne, Gabriel Emery, Marie-Sophie Carrasco'
author = 'Mathieu de Bony de Lavergne, Gabriel Emery, Marie-Sophie Carrasco'

try:
    release = package_version(project)
except PackageNotFoundError:
    release = os.environ.get("READTHEDOCS_VERSION_NAME", "dev")

version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",       # support Google / NumPy style docstrings
    "sphinx.ext.viewcode",       # add links to source
]

autoapi_dirs = ["../../baccmod"]  # path from docs/source to your package
autoapi_keep_files = False
autoapi_add_toctree_entry = False
autoapi_python_class_content = "both"
autoapi_member_order = "bysource"
autoapi_own_page_level = "function"
autoapi_template_dir = "_templates/autoapi"

modindex_common_prefix = ["baccmod."]

nitpick_ignore = [
    ("py:class", "BackgroundCollectionZenith"),
]


# -- Napoleon settings (if you use NumPy or Google style docstrings) -----
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
html_static_path = ['_static']
html_css_files = ["custom.css"]
