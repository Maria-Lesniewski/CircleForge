# How to Add Model Architectures for Testing 
(If you havent read the overview README in the ModelTest section, start there)
(This file covers the fileformat for the json files contained herein)

## Context / STEPS FOR ADDING TESTING FOR NEW MODELS
-New Model Architecture? -> New ref output in ModelTestRunner class.
-New Model Architecture? -> New .json file (FORMAT DESCRIBED HERE), (PATH LISTED IN MODELTESTRUNNER)
-New Model Architecture? -> New reference checkpoint (LISTED IN MODELTESTRUNNER)
-New Model Architecture? -> Check "\_\_LAYER\_TYPES" in ModelTestRunner class for novel PyTorch layers


## JSON FILE FORMAT

Each model implemented in models.model\_options 

Has some "\_\_init\_\_" function implmented defining layers sequenced like 
"""
self.variable\_name = nn.Module(arg) 
"""

Depending on the Module called, args maybe be blank or may correspond to some attribute list.

JSON Files should be formatted as: 

"""
{
 "layers" : [
        "name" : "variable_name",
        "type" : "nn.Module",
        "attributes" : 
           { 
              <input as applicable to the module type>
	    }
] 
}
"""

Where attributes should only appear if there are nonzero attributes to check for the 
particular nn.Module making up the layer. 

For each nn.Module listed under type. The "\_\_LAYER\_TYPES" str : nn.Module dictionary in the ModelTestRunner class must appear. 

