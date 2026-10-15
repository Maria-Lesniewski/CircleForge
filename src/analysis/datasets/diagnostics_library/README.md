# Adding to the diagnostics 

Create a class that inherits from any of the base classes and register it using the DiagnosticsRegistry Decorator factory:
	Rules for registering new files: 
		- The class is in PascalCase, the file is snake_case 
		  (e.g. class: CorrelateFeature is defined in correlate_feature.py)

		- The register key should be the stem name of the file
		  (e.g. correlate_feature.py is registered under "correlate_feature")

	Analysis that don't follow these rules will be flagged as failing in the unit test.
	Likewise, files that are not registered will create failures in the unit test if 
	not added to the list of ignoring files	
