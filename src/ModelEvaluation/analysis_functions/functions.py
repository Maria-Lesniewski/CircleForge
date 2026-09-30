from ..analysis_registry import AnalysisRegistry

@AnalysisRegistry.register_analysis("RMSD_by_category", "special")
def RMSD_by_category(self, mode):
    print("Hello from RMSE_by_category")
