# HOW TO ADD NEW MODEL TESTS

1) Be sure that the model architecture implemented in Modeling.model_options ahs been registered in the Model Factory, your registry key should be the same as the dict keys to be used in step 2 onward

2) Add the appropriate reference checkpoint parameters to the \_checkpoints dict in the ModelTestRunner class

3) Likewise, store an expected output for when a zero tensor is passed. (May need to be reworked for generality in the future)

4) Create a .json file (described within the modelarchitectures folder README.md) that lists the layer names and properties implemented within the model 

5) Review the \_\_LAYER\_TYPES dictionary in ModelTestRunner, any new nn.Modules used should be added with a string key for reading / checking in the tests. Your "type" field will be checked against that dictionary key during tests
