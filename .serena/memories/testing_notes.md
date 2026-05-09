# Testing notes
- `tests/test_import.py` checks that the top-level package imports and that every discovered submodule under `src/hunterFormsBS` imports without errors.
- `tests/conftest.py` auto-discovers the package name from the `src/` directory and walks submodules with `pkgutil.walk_packages`.
- Current visible tests are import-smoke tests, not deep behavioral or numerical model tests.