Problem:

TypeAlias are not protected unless

1. `from __future__ import annotations` is used
2. The `TypeAlias` is manually added to `autodoc_type_aliases`.

[autodoc: Add support for PEP 613 - Explicit Type Alias #8934
](https://github.com/sphinx-doc/sphinx/issues/8934)

This however has the side effect that for `imported-members`, types are no longer properly resolved and hyperlinked.

- We want to import stuff because we use the implemented folder structure.

  ```
  ├── package
  │   ├── __init__.py
  │   ├── _package.py
  ```

- We use the implementer folder structure, because it prevents pollution of namespaces
  - https://stackoverflow.com/questions/57728884/avoiding-module-namespace-pollution-in-python
  - We can fix pollution in `dir(module)` by setting `module.__dir__()` equal to `__all__`.
