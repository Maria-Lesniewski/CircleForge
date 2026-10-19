import pytest 
from analysis.datasets.diagnostics_registry import DiagnosticsRegistry
from pathlib import Path
from utils.io import find_project_root


def test_all_modules_are_registered():
    """ Make sure all relevant files in diagnostic_library are registered correctly """
    
    diagnostic_lib = "src/analysis/datasets/diagnostics_library/"
    path_to_diagnostics_lib = find_project_root() / diagnostic_lib

    analysis_file = [ path.stem for path in path_to_diagnostics_lib.glob("*.py")
            if path.name not in ["__init__.py", "base.py"] ]
    
    for key in analysis_file:
        assert key in DiagnosticsRegistry.available()

def test_registration():
    """ Double check registration mechanism """

    assert "dummy" not in DiagnosticsRegistry.available()
    
    @DiagnosticsRegistry.register_analysis("dummy")
    class DummyDiagnostic:
        pass

    assert "dummy" in DiagnosticsRegistry.available()
