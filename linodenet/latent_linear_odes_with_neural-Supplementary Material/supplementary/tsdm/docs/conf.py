r"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup -------------------------------------------------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import datetime
import os
import sys
from importlib import metadata

import tsdm

os.environ["GENERATING_DOCS"] = "true"
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("./_ext"))

AUTHOR = "Randolf Scholz"
MODULE = "tsdm"
MODULE_DIR = "src/tsdm"
VERSION = metadata.version(MODULE)
YEAR = datetime.datetime.now().year


# region Project Information ------------------------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "tsdm"
# The documented project’s name.

author = "Randolf Scholz"
# The author name(s) of the document. The default value is 'unknown'.

project_copyright = f"{YEAR}, {AUTHOR}"
# A copyright statement in the style '2008, Author Name'.

version = VERSION
# The major project version, used as the replacement for |version|.
# For example, for the Python documentation, this may be something like 2.6.

release = version
# The full project version, used as the replacement for |release| and e.g. in the HTML templates.
# For example, for the Python documentation, this may be something like 2.6.0rc1.

# endregion Project Information ---------------------------------------------------------------------------------------


# region General Configuration ----------------------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "sphinx_toolbox.more_autodoc.typehints",
    # "sphinx_toolbox.more_autodoc.typevars",
    # "sphinx_toolbox.more_autodoc.genericalias",
    # Sphinx builtin extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    # 1st party extensions
    "signatures",
    # 3rd party extensions
    # "autoapi.extension",
    "sphinx_copybutton",
    # "sphinx_math_dollar",
    # "sphinx_autodoc_typehints",
]
# Add any Sphinx extension module names here, as strings. They can be extensions coming with Sphinx
# (named 'sphinx.ext.*') or your custom ones.


root_doc = "index"
# The document name of the “root” document, that is, the document that contains the root toctree directive.
# Default is 'index'.

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# A list of glob-style patterns that should be excluded when looking for source files. They are matched against the
# source file names relative to the source directory, using slashes as directory separators on all platforms.

include_patterns = ["**"]
# A list of glob-style patterns [1] that are used to find source files. They are matched against the source file names
# relative to the source directory, using slashes as directory separators on all platforms.
# The default is **, meaning that all files are recursively included from the source directory.

templates_path = ["_templates"]
# A list of paths that contain extra templates (or templates that overwrite builtin/theme-specific templates).
# Relative paths are taken as relative to the configuration directory.

rst_epilog = ""
# A string of reStructuredText that will be included at the end of every source file that is read. This is a possible
# place to add substitutions that should be available in every file (another being rst_prolog).
# An example: rst_epilog = """.. |psf| replace:: Python Software Foundation"""

rst_prolog = ""
# A string of reStructuredText that will be included at the beginning of every source file that is read.
# This is a possible place to add substitutions that should be available in every file (another being rst_epilog).
# An example: rst_epilog = """.. |psf| replace:: Python Software Foundation"""

primary_domain = "py"
# The name of the default domain. Can also be None to disable a default domain. The default is 'py'.

default_role = "py:obj"
# The name of a reST role (builtin or Sphinx extension) to use as the default role, that is, for text marked up
# `like this`. This can be set to 'py:obj' to make `filter` a cross-reference to the Python function “filter”.
# The default is None, which doesn't reassign the default role.

keep_warnings = False
# If true, keep warnings as “system message” paragraphs in the built documents. Regardless of this setting,
# warnings are always written to the standard error stream when sphinx-build is run. The default is False.

suppress_warnings = []
# A list of warning types to suppress arbitrary warning messages.


needs_sphinx = "5.1"
# If set to a major.minor version string like '1.1',
# Sphinx will compare it with its version and refuse to build if it is too old. Default is no requirement.

needs_extensions = {}
# This value can be a dictionary specifying version requirements for extensions in extensions, e.g.
# needs_extensions = {'sphinxcontrib.something': '1.5'}. The version strings should be in the form major.minor.
# Requirements do not have to be specified for all extensions, only for those you want to check.

nitpicky = False
# If true, Sphinx will warn about all references where the target cannot be found. Default is False.
# You can activate this mode temporarily using the -n command-line switch.

smartquotes = True
# If true, the Docutils Smart Quotes transform, originally based on SmartyPants (limited to English) and currently
# applying to many languages, will be used to convert quotes and dashes to typographically correct entities.
# Default: True.

pygments_style = "default"
# The style name to use for Pygments highlighting of source code.
# If not set, either the theme’s default style or 'sphinx' is selected for HTML output.

add_function_parentheses = False
# A boolean that decides whether parentheses are appended to function and method role text
# (e.g. the content of :func:`input`) to signify that the name is callable. Default is True.

add_module_names = False
# A boolean that decides whether module names are prepended to all object names
# (for object types where a “module” of some kind is defined), e.g. for py:function directives. Default is True.

show_authors = True
# A boolean that decides whether codeauthor and sectionauthor directives produce any output in the built files.

python_use_unqualified_type_names = True
# If true, suppress the module name of the python reference if it can be resolved. The default is False

# endregion General Configuration -------------------------------------------------------------------------------------


# region HTML Configuration ------------------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_style=???
# The style sheet to use for HTML pages. A file of that name must exist either in Sphinx’s static/ path,
# or in one of the custom paths given in html_static_path. Default is the stylesheet given by the selected theme.
# If you only want to add or override a few things compared to the theme’s stylesheet,
# use CSS @import to import the theme’s stylesheet.

html_title = f"{MODULE} {VERSION}"
# The “title” for HTML documentation generated with Sphinx’s own templates.
# This is appended to the <title> tag of individual pages, and used in the navigation bar as the “topmost” element.
# It defaults to '<project> v<revision> documentation'.

html_short_title = MODULE
# A shorter “title” for the HTML docs. This is used for links in the header and in the HTML Help docs.
# If not given, it defaults to the value of html_title.

html_baseurl = ""
# The base URL which points to the root of the HTML documentation.
# It is used to indicate the location of document using The Canonical Link Relation. Default: ''.

html_context = {}
# A dictionary of values to pass into the template engine’s context for all pages.
# Single values can also be put in this dictionary using the -A command-line option of sphinx-build.

html_logo = None
# If given, this must be the name of an image file (path relative to the configuration directory)
# that is the logo of the docs, or URL that points an image file for the logo. It is placed at the top of the sidebar;
# its width should therefore not exceed 200 pixels. Default: None.


html_favicon = None
# If given, this must be the name of an image file (path relative to the configuration directory) that is the favicon
# of the docs, or URL that points an image file for the favicon.
# Modern browsers use this as the icon for tabs, windows and bookmarks.
# It should be a Windows-style icon file (.ico), which is 16x16 or 32x32 pixels large. Default: None.

html_static_path = ["_static"]
# A list of paths that contain custom static files (such as style sheets or script files).
# Relative paths are taken as relative to the configuration directory. They are copied to the output’s _static
# directory after the theme’s static files, so a file named default.css will overwrite the theme’s default.css.

html_extra_path = []
# A list of paths that contain extra files not directly related to the documentation, such as robots.txt or .htaccess.
# Relative paths are taken as relative to the configuration directory. They are copied to the output directory.
# They will overwrite any existing file of the same name.

html_use_smartypants = True
# If true, quotes and dashes are converted to typographically correct entities. Default: True.

html_permalinks = True
# If true, Sphinx will add “permalinks” for each heading and description environment. Default: True.

html_permalinks_icon = "§"
# A text for permalinks for each heading and description environment. HTML tags are allowed. Default: a paragraph sign;

html_sidebars = {}
# Custom sidebar templates, must be a dictionary that maps document names to template names.

# TODO: Add missing configuration options.

# endregion HTML Configuration ---------------------------------------------------------------------------------


# region Theme Configuration -----------------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/theming.html

# extensions.append("sphinx_typo3_theme")

# piccolo_theme, karma_sphinx_theme, sphinx_rtd_theme, pydata_sphinx_theme, sphinx_typo3_theme
html_theme = "pydata_sphinx_theme"
# The “theme” that the HTML output should use. See the section about theming. The default is 'alabaster'.

html_theme_path = []
# A list of paths that contain custom themes, either as subdirectories or as zip files.
# Relative paths are taken as relative to the configuration directory.

# html_theme_options = {
#     # TOCTREE OPTIONS
#     "collapse_navigation": False,
#     # With this enabled, navigation entries are not expandable – the [+] icons next to each entry are removed.
#     # Default: True
#     "sticky_navigation": True,
#     # Scroll the navigation with the main page content as you scroll the page.
#     # Default: True
#     "navigation_depth": 4,
#     # The maximum depth of the table of contents tree. Set this to -1 to allow unlimited depth.
#     # Default: 4
#     "includehidden": True,
#     # Specifies if the navigation includes hidden table(s) of contents – that is,
#     # any toctree directive that is marked with the :hidden: option.
#     # Default: True
#     "titles_only": True,
#     # When enabled, page subheadings are not included in the navigation.
#     # Default: False
#     # MISCELLANEOUS OPTIONS
#     "analytics_id": "",
#     # If specified, Google Analytics’ gtag.js is included in your pages.
#     # Set the value to the ID provided to you by google (like UA-XXXXXXX or G-XXXXXXXXXX).
#     "analytics_anonymize_ip": False,
#     # Anonymize visitor IP addresses in Google Analytics.
#     # Default: False
#     "display_version": True,
#     # If True, the version number is shown at the top of the sidebar.
#     # Default: True
#     "logo_only": False,
#     # Only display the logo image, do not display the project name at the top of the sidebar
#     # Default: False
#     "prev_next_buttons_location": "bottom",
#     # Location to display Next and Previous buttons. This can be either bottom, top, both , or None.
#     # Default: "bottom"
#     "style_external_links": False,
#     # Add an icon next to external links.
#     # Default: False
#     "vcs_pageview_mode": "blob",
#     # Changes how to view files when using display_github, display_gitlab, etc. When using GitHub or GitLab
#     # this can be: blob (default), edit, or raw. On Bitbucket, this can be either: view (default) or edit.
#     # Default: "blob" or "view"
#     "style_nav_header_background": r"#2980B9",
#     # Changes the background of the search area in the navigation bar.
#     # The value can be anything valid in a CSS background property.
#     # Default: "#2980B9"
# }
# A dictionary of options that influence the look and feel of the selected theme. These are theme-specific.

# endregion Theme Configuration --------------------------------------------------------------------------------


# region sphinx-autoapi configuration ---------------------------------------------------------------------------------
# https://github.com/readthedocs/sphinx-autoapi

autoapi_dirs = [f"../{MODULE_DIR}"]
# Paths (relative or absolute) to the source code that you wish to generate your API documentation from.

autoapi_type = "python"
# Set the type of files you are documenting. This depends on the programming language that you are using.
# Default: "python"

autoapi_template_dir = "_templates/autoapi"
# A directory that has user-defined templates to override our default templates. The path can either be absolute,
# or relative to the source directory of your documentation files. A path relative to where sphinx-build is run is
# allowed for backwards compatibility only and will be removed in a future version.
# Default: ""

autoapi_file_patterns = ["*.py", "*.pyi"]
# A list containing the file patterns to look for when generating documentation.
# Patterns should be listed in order of preference. For example, if autoapi_file_patterns is set to the default value,
# and a .py file and a .pyi file are found, then the .py will be read.

autoapi_generate_api_docs = True
# Whether to generate API documentation. If this is False, documentation should be generated though the Directives.
# Default: True

autoapi_options = [
    # "members",
    # "special-members",
    "imported-members",
    # "undoc-members",
    # "private-members",
    # "show-inheritance",
    # "show-module-summary",
]
# Options for display of the generated documentation.
# Default: [ 'members', 'undoc-members', 'private-members', 'show-inheritance', 'show-module-summary',
# 'special-members', 'imported-members', ]

autoapi_ignore = ["*migrations*"]
# A list of patterns to ignore when finding files. The defaults by language are:
# Default = ['*migrations*']

autoapi_root = "autoapi"
# Path to output the generated AutoAPI files into, including the generated index page.
# This path must be relative to the source directory of your documentation files.
# This can be used to place the generated documentation anywhere in your documentation hierarchy.
# Default: "autoapi"

autoapi_add_toctree_entry = False
# Whether to insert the generated documentation into the TOC tree. If this is False, the default AutoAPI index page
# is not generated, and you will need to include the generated documentation in a TOC tree entry yourself.
# Default: True

autoapi_member_order = "groupwise"
# The order to document members. This option can have the following values:
# alphabetical: Order members by their name, case sensitively.
# bysource: Order members by the order that they were defined in the source code.
# groupwise: Order members by their type then alphabetically, ordering the types as follows:
# Submodules and subpackages, Attributes, Exceptions, Classes, Functions, and Methods.
# Default: bysource

autoapi_python_class_content = "both"
# Which docstring to insert into the content of a class.
# If the class does not have an __init__ or the __init__ docstring is empty and
# the class defines a __new__ with a docstring, the __new__ docstring is used instead of the __init__ docstring.
# Default: "class"

autoapi_python_use_implicit_namespaces = True
# This changes the package detection behaviour to be compatible with PEP 420,
# but directories in autoapi_dirs are no longer searched recursively for packages. Instead, when this is True,
# autoapi_dirs should point directly to the directories of implicit namespaces and the directories of packages.
# Default: False

autoapi_prepare_jinja_env = None
# A callback that is called shortly after the Jinja environment is created.
# It passed the Jinja environment for editing before template rendering begins.
# Default: None

autoapi_keep_files = True
# Keep the AutoAPI generated files on the filesystem after the run.
# Useful for debugging or transitioning to manual documentation.
# Keeping files will also allow AutoAPI to use incremental builds. Providing none of the source files have changed,
# AutoAPI will skip parsing the source code and regenerating the API documentation.
# Default: False

# endregion sphinx-autoapi configuration ------------------------------------------------------------------------------


# region sphinx.ext.autodoc configuration -----------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#directive-autoclass

autoclass_content = "both"
# This value selects what content will be inserted into the main body of an autoclass directive.
# The possible values are: (default="class")
# "class"
# Only the class’ docstring is inserted. This is the default.
# You can still document __init__ as a separate method using automethod or the members option to autoclass.
# "both"
# Both the class’ and the __init__ method’s docstring are concatenated and inserted.
# "init"
# Only the __init__ method’s docstring is inserted.
# If the class has no __init__ method or if the __init__ method’s docstring is empty,
# but the class has a __new__ method’s docstring, it is used instead.

autodoc_class_signature = "mixed"
# This value selects how the signature will be displayed for the class defined by autoclass directive.
# The possible values are: (default="mixed")
# "mixed"
# Display the signature with the class name.
# "separated"
# Display the signature as a method.

autodoc_member_order = "groupwise"
# This value selects if automatically documented members are sorted alphabetical (value 'alphabetical'),
# by member type (value 'groupwise') or by source order (value 'bysource'). The default is alphabetical.
# Note that for source order, the module must be a Python module with the source code available.

autodoc_default_flags = []
# This value is a list of autodoc directive flags that should be automatically applied to all autodoc directives.
# The supported flags are 'members', 'undoc-members', 'private-members', 'special-members', 'inherited-members',
# 'show-inheritance', 'ignore-module-all' and 'exclude-members'.
# autodoc_default_options = {}
# The default options for autodoc directives. They are applied to all autodoc directives automatically.
# It must be a dictionary which maps option names to the values. For example:
#
autodoc_default_options = {
    # 'members': 'var1, var2',
    # 'member-order': 'groupwise',
    # 'special-members': '__init__',
    "undoc-members": False,
    # 'exclude-members': '__weakref__'
}
# Setting None or True to the value is equivalent to giving only the option name to the directives.
# The supported options are 'members', 'member-order', 'undoc-members', 'private-members', 'special-members',
# 'inherited-members', 'show-inheritance', 'ignore-module-all', 'imported-members', 'exclude-members' and
# 'class-doc-from'.

autodoc_docstring_signature = True
# Functions imported from C modules cannot be introspected, and therefore the signature for such functions cannot be
# automatically determined. However, it is an often-used convention to put the signature into the first line of the
# function’s docstring.
# If this boolean value is set to True (which is the default), autodoc will look at the first line of the docstring for
# functions and methods, and if it looks like a signature, use the line as the signature and remove it from the
# docstring content.
# autodoc will continue to look for multiple signature lines, stopping at the first line that does not look like a
# signature. This is useful for declaring overloaded function signatures.

autodoc_mock_imports = []
# This value contains a list of modules to be mocked up.
# This is useful when some external dependencies are not met at build time and break the building process.
# You may only specify the root package of the dependencies themselves and omit the submodules:

autodoc_typehints = "both"
# This value controls how to represent typehints. The setting takes the following values:
# 'signature' – Show typehints in the signature (default)
# 'description' – Show typehints as content of the function or method The typehints of overloaded
#                 functions or methods will still be represented in the signature.
# 'none' – Do not show typehints
# 'both' – Show typehints in the signature and as content of the function or method
# Overloaded functions or methods will not have typehints included in the description
# because it is impossible to accurately represent all possible overloads as a list of parameters.

autodoc_typehints_description_target = "documented"
# This value controls whether the types of undocumented parameters and return values are
# documented when autodoc_typehints is set to description. The default value is "all", meaning that
# the types are documented for all parameters and return values, whether they are documented or not.
# When set to "documented", types will only be documented for a parameter or a return value that is
# already documented by the docstring.

autodoc_type_aliases = {
    # tsdm.utils.strings.AliasType : '~tsdm.utils.strings.AliasType',
    "AliasType": "~tsdm.utils.strings.AliasType",
    "Path": "pathlib.Path",
    # torch
    "Tensor": "~torch.Tensor",
    "nn.Module": "~torch.nn.Module",
    "SummaryWriter": "~torch.utils.tensorboard.writer.SummaryWriter",
    # numpy
    "ArrayLike": "~numpy.typing.ArrayLike",
    "datetime64": "~numpy.datetime64",
    "timedelta64": "~numpy.timedelta64",
    "integer": "~numpy.integer",
    "floating": "~numpy.floating",
    # pandas
    "NA": "~pandas.NA",
    "NaT": "~pandas.NaT",
    "DataFrame": "~pandas.DataFrame",
    "Series": "`~pandas.Series`",
    "Index": "~pandas.Index",
    "MultiIndex": "~pandas.MultiIndex",
    "CategoricalIndex": "~pandas.CategoricalIndex",
    "TimedeltaIndex": "~pandas.TimedeltaIndex",
    "DatetimeIndex": "~pandas.DatetimeIndex",
    "Categorical": "~pandas.Categorical",
    "get_dummies": "~pandas.get_dummies",
    # xarray
    "DataArray": "~xarray.DataArray",
    "Dataset": "~xarray.Dataset",
    "Variable": "~xarray.Variable",
}
# A dictionary for users defined type aliases that maps a type name to the full-qualified object name.
# It is used to keep type aliases not evaluated in the document. Defaults to empty ({}).
# The type aliases are only available if your program enables Postponed Evaluation of Annotations (PEP 563)
# feature via from __future__ import annotations.
# autodoc_type_aliases |= tsdm.utils.system.get_napoleon_type_aliases(typing)
# # recursively napoleon_type_aliases for tsdm classes / functions.
# autodoc_type_aliases |= tsdm.utils.system.get_napoleon_type_aliases(abc)
# # recursively napoleon_type_aliases for tsdm classes / functions.
# autodoc_type_aliases |= tsdm.utils.system.get_napoleon_type_aliases(torch)
# autodoc_type_aliases |= tsdm.utils.system.get_napoleon_type_aliases(torch.utils)
autodoc_type_aliases |= tsdm.utils.system.get_napoleon_type_aliases(tsdm)
print(autodoc_type_aliases)
# # recursively napoleon_type_aliases for tsdm classes / functions.

autodoc_typehints_format = "short"
# This value controls the format of typehints. The setting takes the following values:
# 'fully-qualified' – Show the module name and its name of typehints
# 'short' – Suppress the leading module names of the typehints (ex. io.StringIO -> StringIO)

autodoc_preserve_defaults = True
# If True, the default argument values of functions will be not evaluated on generating document.
# It preserves them as is in the source code.

autodoc_warningiserror = True
# This value controls the behavior of sphinx-build -W during importing modules. If False is given,
# autodoc forcedly suppresses the error if the imported module emits warnings. By default, True.

autodoc_inherit_docstrings = True
# This value controls the docstrings inheritance. If set to True the docstring for classes or methods,
# if not explicitly set, is inherited from parents. The default is True.

# endregion sphinx.ext.autodoc configuration --------------------------------------------------------------------------


# region sphinx.ext.autosectionlabel configuration --------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html

autosectionlabel_prefix_document = True
# True to prefix each section label with the name of the document it is in, followed by a colon.
# For example, index:Introduction for a section called Introduction that appears in document index.rst.
# Useful for avoiding ambiguity when the same section heading appears in different documents.
autosectionlabel_maxdepth = None
# If set, autosectionlabel chooses the sections for labeling by its depth.
# For example, when set 1 to autosectionlabel_maxdepth, labels are generated only for top level sections,
# and deeper sections are not labeled. It defaults to None (disabled).

# endregion sphinx.ext.autosectionlabel configuration -----------------------------------------------------------------


# region sphinx.ext.autosummary configuration -------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html

autosummary_context = {}
# A dictionary of values to pass into the template engine’s context for autosummary stubs files.

autosummary_generate = True
# Boolean indicating whether to scan all found documents for autosummary directives,
# and to generate stub pages for each. It is enabled by default.

autosummary_generate_overwrite = True
# If true, autosummary overwrites existing files by generated stub pages. Defaults to true (enabled).

autosummary_mock_imports = []
# This value contains a list of modules to be mocked up. See autodoc_mock_imports for more details.
# It defaults to autodoc_mock_imports.

autosummary_imported_members = False
# A boolean flag indicating whether to document classes and functions imported in modules. Default is False

autosummary_ignore_module_all = False
# If False and a module has the __all__ attribute set,
# autosummary documents every member listed in __all__ and no others.
# Note that if an imported member is listed in __all__, it will be documented regardless of the value of
# autosummary_imported_members. To match the behaviour of from module import *, set autosummary_ignore_module_all to
# False and autosummary_imported_members to True.
# Default is True

autosummary_filename_map = {}
# A dict mapping object names to filenames. This is necessary to avoid filename conflicts where multiple objects
# have names that are indistinguishable when case is ignored, on file systems where filenames are case-insensitive.

# endregion sphinx.ext.autosummary configuration ----------------------------------------------------------------------


# region sphinx.ext.intersphinx configuration -------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    # "numba": ("https://numba.pydata.org/numba-doc/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "python": ("https://docs.python.org/3.10/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    # "tsdm": ("_build_old/html", None),
    # "xarray": ("https://xarray.pydata.org/en/stable/", None),
}
# This config value contains the locations and names of other projects that should be linked to
# in this documentation.
# Relative local paths for target locations are taken as relative to the base of the
# built documentation, while relative local paths for inventory locations are taken as
# relative to the source directory.
# When fetching remote inventory files, proxy settings will be read from the
# $HTTP_PROXY environment variable.

intersphinx_cache_limit = 5
# The maximum number of days to cache remote inventories. The default is 5,
# meaning five days. Set this to a negative value to cache inventories for unlimited time.

intersphinx_timeout = 2
# The number of seconds for timeout. The default is None, meaning do not time out.

intersphinx_disabled_reftypes = ["std:doc"]
# When a cross-reference without an explicit inventory specification is being resolved by
# intersphinx, skip resolution if it matches one of the specifications in this list.
# The default value is ['std:doc'].

# endregion sphinx.ext.intersphinx configuration ----------------------------------------------------------------------


# region sphinx.ext.mathjax configuration -----------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/math.html#module-sphinx.ext.mathjax

# mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
# The path to the JavaScript file to include in the HTML files in order to load MathJax.
# The default is the https:// URL that loads the JS files from the jsdelivr Content Delivery Network.
# See the MathJax Getting Started page for details. If you want MathJax to be available offline or without including
# resources from a third-party site, you have to download it and set this value to a different path.

mathjax_options = {}
# The options to script tag for mathjax.
# The default is empty {}.

mathjax3_config = {
    "loader": {"load": [r"[tex]/physics"]},
    "tex": {
        "inlineMath": [[r"\(", r"\)"]],
        "displayMath": [[r"\[", r"\]"]],
        "packages": {"[+]": ["physics"]},
        "macros": {
            "argmax": r"\operatorname*{arg\,max}",
            "argmin": r"\operatorname*{arg\,min}",
            "diag": r"\operatorname{diag}",
            "rank": r"\operatorname{rank}",
            "relu": r"\operatorname{ReLU}",
            "tr": r"\operatorname{tr}",
        },
    },
}
# The configuration options for MathJax v3 (which is used by default).

mathjax2_config = {}
# The configuration options for MathJax v2
# The default is empty {}.

mathjax_config = {
    "tex2jax": {
        "inlineMath": [[r"\(", r"\)"]],
        "displayMath": [[r"\[", r"\]"]],
    },
}  # Former name of mathjax2_config.

# endregion sphinx.ext.mathjax configuration --------------------------------------------------------------------------


# region sphinx.ext.napoleon configuration ----------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_google_docstring = True
# True to parse Google style docstrings.
# False to disable support for Google style docstrings.
# Defaults to True.

napoleon_numpy_docstring = True
# True to parse NumPy style docstrings.
# False to disable support for NumPy style docstrings.
# Defaults to True.

napoleon_include_init_with_doc = True
# True to list __init___ docstrings separately from the class docstring.
# False to fall back to Sphinx’s default behavior,
# which considers the __init___ docstring as part of the class documentation.
# Defaults to False.

napoleon_include_private_with_doc = False
# True to include private members (like _membername) with docstrings in the documentation.
# False to fall back to Sphinx’s default behavior.
# Defaults to False.

napoleon_include_special_with_doc = True
# True to include special members (like __membername__) with docstrings in the documentation.
# False to fall back to Sphinx’s default behavior.
# Defaults to True.

napoleon_use_admonition_for_examples = True
# True to use the .. admonition:: directive for the Example and Examples sections.
# False to use the .. rubric:: directive instead.
# One may look better than the other depending on what HTML theme is used.
# Defaults to False.

napoleon_use_admonition_for_notes = True
# True to use the .. admonition:: directive for Notes sections.
# False to use the .. rubric:: directive instead.
# Defaults to False.

napoleon_use_admonition_for_references = True
# True to use the .. admonition:: directive for References sections.
# False to use the .. rubric:: directive instead.
# Defaults to False.

napoleon_use_ivar = True
# True to use the :ivar: role for instance variables.
# False to use the .. attribute:: directive instead.
# Defaults to False.

napoleon_use_param = True
# True to use a :param: role for each function parameter.
# False to use a single :parameters: role for all the parameters.
# Defaults to True.

napoleon_use_keyword = True
# True to use a :keyword: role for each function keyword argument.
# False to use a single :keyword arguments: role for all the keywords.
# Defaults to True.

napoleon_use_rtype = True
# True to use the :rtype: role for the return type.
# False to output the return type inline with the description.
# Defaults to True.

napoleon_preprocess_types = True
# True to convert the type definitions in the docstrings as references.
# Defaults to True.

napoleon_type_aliases = {
    "Path": "~pathlib.Path",
    # torch
    "torch": "`torch`",
    "Tensor": "~torch.Tensor",
    "nn.Module": "~torch.nn.Module",
    "SummaryWriter": "~torch.utils.tensorboard.writer.SummaryWriter",
    # numpy
    "ArrayLike": "~numpy.typing.ArrayLike",
    "datetime64": "~numpy.datetime64",
    "timedelta64": "~numpy.timedelta64",
    "integer": "~numpy.integer",
    "floating": "~numpy.floating",
    # pandas
    "NA": "~pandas.NA",
    "NaT": "~pandas.NaT",
    "DataFrame": "~pandas.DataFrame",
    "Series": "~pandas.Series",
    "Index": "~pandas.Index",
    "MultiIndex": "~pandas.MultiIndex",
    "CategoricalIndex": "~pandas.CategoricalIndex",
    "TimedeltaIndex": "~pandas.TimedeltaIndex",
    "DatetimeIndex": "~pandas.DatetimeIndex",
    "Categorical": "~pandas.Categorical",
    # xarray
    "DataArray": "~xarray.DataArray",
    "Dataset": "~xarray.Dataset",
    "Variable": "~xarray.Variable",
}
# A mapping to translate type names to other names or references. Works only when napoleon_use_param = True.
# Defaults to None.

# napoleon_type_aliases |= tsdm.utils.system.get_napoleon_type_aliases(typing)
# # recursively napoleon_type_aliases for tsdm classes / functions.
# napoleon_type_aliases |= tsdm.utils.system.get_napoleon_type_aliases(abc)
# # recursively napoleon_type_aliases for tsdm classes / functions.
napoleon_type_aliases |= tsdm.utils.system.get_napoleon_type_aliases(tsdm)
# recursively napoleon_type_aliases for tsdm classes / functions.

napoleon_attr_annotations = True
# True to allow using PEP 526 attributes annotations in classes. If an attribute is documented in the docstring without
# a type and has an annotation in the class body, that type is used.

napoleon_custom_sections = ["Test-Metric", "Evaluation Protocol", "Paper", "Results"]
# Add a list of custom sections to include, expanding the list of parsed sections. Defaults to None.

# endregion sphinx.ext.napoleon configuration -------------------------------------------------------------------------


# region sphinx.ext.todo configuration --------------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html

todo_include_todos = False
# If this is True, todo and todolist produce output, else they produce nothing. The default is False.
todo_emit_warnings = False
# If this is True, todo emits a warning for each TODO entries. The default is False.
todo_link_only = False
# If this is True, todolist produce output without file path and line, The default is False.

# endregion sphinx.ext.todo configuration -----------------------------------------------------------------------------


# region sphinx.ext.viewcode configuration ----------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/viewcode.html

viewcode_follow_imported_members = True
# If this is True, viewcode extension will emit viewcode-follow-imported event to resolve the name of the module by
# other extensions. The default is True.

viewcode_enable_epub = False
# If this is True, viewcode extension is also enabled even if you use epub builders.
# This extension generates pages outside toctree, but this is not preferred as epub format.
# Until 1.4.x, this extension is always enabled. If you want to generate epub as same as 1.4.x, you should set True,
# but epub format checker’s score becomes worse.
# The default is False.

# endregion sphinx.ext.viewcode configuration -------------------------------------------------------------------------


# region sphinx_math_dollar configuration ------------------------------------------------------------------------------

# https://www.sympy.org/sphinx-math-dollar/#configuration
# from sphinx_math_dollar import NODE_BLACKLIST
# from docutils.nodes import header
# from sphinx.addnodes import pending_xref_condition
# math_dollar_node_blacklist = NODE_BLACKLIST + (header, pending_xref_condition)


# endregion sphinx_math_dollar configuration ---------------------------------------------------------------------------


# -- end of configuration ---------------------------------------------------------------------------------------------


# extensions.append('sphinx_automodapi.automodapi')
# extensions.append('sphinx_automodapi.smart_resolver')
# numpydoc_show_class_members = False
