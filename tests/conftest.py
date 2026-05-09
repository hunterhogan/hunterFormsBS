"""Zero-configuration conftest for src-layout projects."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
import importlib
import pkgutil
import pytest
import warnings

def _discover_package_name() -> str:
	project_root = Path(__file__).resolve().parent.parent
	src = project_root / 'src'
	if not src.is_dir():
		msg = f"'src' directory not found at {src}"
		raise FileNotFoundError(msg)

	packages = [d.name for d in src.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name != '__pycache__']
	if not packages:
		msg = f'No Python packages found in {src}'
		raise FileNotFoundError(msg)

	if len(packages) > 1:
		warnings.warn(f"Multiple packages found in src: {packages}. Using '{packages[0]}' as the main package.", stacklevel=2)
	return packages[0]


@pytest.fixture(scope='session')
def package_name() -> str:
	return _discover_package_name()


@pytest.fixture(scope='session')
def package(package_name: str) -> ModuleType:
	"""The imported top-level package."""
	return importlib.import_module(package_name)


@pytest.fixture(scope='session')
def all_module_names(package_name: str) -> list[str]:
	"""All module names in the package (including subpackages)."""
	pkg = importlib.import_module(package_name)
	modules = [package_name]
	prefix = package_name + '.'
	for _idk, moduleName, _lame in pkgutil.walk_packages(pkg.__path__, prefix=prefix):
		modules.append(moduleName)
	return modules
