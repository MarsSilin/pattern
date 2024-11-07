# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import os.path
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
# import pathlib


autodoc_mock_imports = [
    "numpy",
    "pandas",
    "click",
    "tslearn",
    "apimoex",
    "optuna",
    "matplotlib",
]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pattern or coincidence"
copyright = "2023, pattern fans"
author = "pattern fans"
release = "0.0.1"


# sys.path.insert(0, pathlib.Path('D:/Programs/jupyter/patterns/patterns/
# pattern_or_coincidence').parents[2].resolve().as_posix())
# sys.path.append('D:/Programs/jupyter/patterns/patterns/pattern_or_coincidence')
cdir = os.path.dirname(os.getcwd())
cdir.replace(os.sep, "/")
sys.path.append(cdir + "/src/data")
sys.path.append(cdir + "/src/models")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
