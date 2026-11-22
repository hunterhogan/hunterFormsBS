"""Zero-configuration conftest for flat-layout projects."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import importlib
import pkgutil
import pytest
import warnings

if TYPE_CHECKING:
	from types import ModuleType

def _discover_package_name() -> str:
	pathTests = Path(__file__).resolve().parent
	project_root = pathTests.parent

	packages = sorted(d.name for d in project_root.iterdir() if d.is_dir() and d != pathTests and (d / '__init__.py').is_file())
	if not packages:
		msg = f'No Python packages found in {project_root}'
		raise FileNotFoundError(msg)

	if len(packages) > 1:
		warnings.warn(f"Multiple packages found in {project_root}: {packages}. Using '{packages[0]}' as the main package.", stacklevel=2)
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
