import pytest 
from StatisticalAnalysis.ModelEvaluation.analysis_registry import AnalysisRegistry
from pathlib import Path
from utils.io import find_project_root


def test_all_modules_are_registered():
    """ Make sure all relevant files in diagnostic_library are registered correctly """
    
    diagnostic_lib = "src/StatisticalAnalysis/ModelEvaluation/analysis_functions/"
    path_to_diagnostics_lib = find_project_root() / diagnostic_lib

    analysis_file = [ path.stem for path in path_to_diagnostics_lib.glob("*.py")
            if path.name not in ["__init__.py", "base.py"] ]
   
    for key in analysis_file:
        assert key in AnalysisRegistry.total_available()

def test_basic_registration():
    """ Check registration mechanism goes into regular list"""
    
    assert "dummy" not in AnalysisRegistry.available()

    @AnalysisRegistry.register_analysis("dummy")
    class DummyAnalysis:
        pass

    assert "dummy" in AnalysisRegistry.available()


def test_special_registration():
    """ Check special keyword adds to seperate list """ 
    assert "dummy2" not in AnalysisRegistry.available()
    assert "dummy2" not in AnalysisRegistry.special_available()
    assert "dummy2" not in AnalysisRegistry.total_available()

    @AnalysisRegistry.register_analysis("dummy2", "special")
    class Dummy2:
        pass

    assert "dummy2" not in AnalysisRegistry.available()
    assert "dummy2" in AnalysisRegistry.total_available()
    assert "dummy2" in AnalysisRegistry.special_available()
