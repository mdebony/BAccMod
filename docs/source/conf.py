# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'BAccMod'
copyright = '2025, Mathieu de Bony de Lavergne, Gabriel Emery, Marie-Sophie Carrasco'
author = 'Mathieu de Bony de Lavergne, Gabriel Emery, Marie-Sophie Carrasco'
release = '0.4.0_dev'

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


# If your package is not on sys.path by default, insert it. Example assuming docs/ is sibling to package:
import os
import sys
sys.path.insert(0, os.path.abspath(".."))  # adjust if needed

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
