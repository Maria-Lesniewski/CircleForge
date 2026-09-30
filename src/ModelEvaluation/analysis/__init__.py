from ..analysis_registry import AnalysisRegistry
from pkgutil import iter_modules
from importlib import import_module

for module in iter_modules(__path__):
    if not module.name.startswith("_"):
        import_module(f"{__name__}.{module.name}")
