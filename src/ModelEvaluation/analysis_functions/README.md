# Adding New Functions (methods as classes) for Availability in the AnalysisRunner 

### Motivation
The functions in this folder are designed to avoid iterating over the dataset multiple times/
Running model inference multiple times

They are registered with an analysis runner which actually calls/executes them in class as object fashion

### Procedure

1) Review the base classes in base.py
	- BasicFunction -> Data task can be completed while data is streaming
	- StatefulFunction -> Data task requires aggregrate knowledge that should be
			      finalized post streaming
	- ConfigurableFunction -> Data task requires some extra input to be streamable

2) Create a file (within this folder) with your new analysis name that imports the corresponding base class and the AnalysisRegistry

3) Create a class with your new analysis name as the class, decorate it with the AnalysisRegistry.register function:
	- Pass the string name (usually same as filename) for which the function will be registered, by default you register it = it will appear in the register
	- If the function should only be available under certain conditions, pass a second keyword to mark it in the special section 
	  of AnalysisRegistry

4) Implement abstractmethods from the base class as classmethods using the classmethods wrapper, 
   add the attributes as suggested from the base class. 

5) Congratulations! If you follow all these steps, the function will be callable using AnalysisRegistry.available(), AnalysisRegistry.special\_available() etc.
