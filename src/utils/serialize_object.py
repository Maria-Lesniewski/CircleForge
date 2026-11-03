# This is a handy function for serializing objects to a json printable format.
# I am not sure it really belonds in io or logger, so I'm giving it its own module

import torch
import torch.nn as nn
import torch.optim as optim
import json

def serialize_object(obj):
    """Convert an object into JSON-serializable data."""
    
    attributes_dict = {}
    for attr_name, attr_value in vars(obj).items():
        if hasattr(attr_value, "to_dict"):
            attributes_dict[attr_name] = attr_value.to_dict()

        # For now handle optimizer explicitly
        elif isinstance(attr_value, optim.Optimizer):
            attributes_dict[attr_name] = {
                    "__class__": attr_value.__class__.__name__,
                    "param_groups" : [
                        {
                            key: value
                            for key, value in group.items()
                            if key != "params"
                            } 
                        for group in attr_value.param_groups],
                    }
        else: 
            try:
                json.dumps(attr_value)
                attributes_dict[attr_name] = attr_value
            except (TypeError, ValueError):
                attributes_dict[attr_name] = f"<{attr_value.__class__.__name__}>"

    object_dict = {obj.__class__.__name__ : attributes_dict}
    return object_dict
