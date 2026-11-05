import torch
import torch.nn as nn
from typing import Any

class NNModuleDeviceAuditor(nn.Module):
    """
    This is a class sole built to help me catch some of the
    mistakes associated with building a PyTorch workflow 
    while only having CPU access
    (e.g. data/moduels not on same devices when multiple are avail)
    """
    def __init__(self, config: dict[str,Any]=None):
        super().__init__()

        self.audit = {
                "device_configured" : False,
                "device_source" : None,
                "device" : None,
                "device_calls" : [],
                }

        # Dummy weights for inspecting placement
        self.weight = nn.Parameter(torch.zeros(1))
        
        if config is not None:
            device = config.get("device", None)

    def to(self, *args, **kwargs):
        device = kwargs.get("device", None)

        if device is None and args: 
            if isinstance(args[0], (str, torch.device)):
                device = args[0]
        
        if device is not None:
            self.audit["device"] = str(device)
            self.audit["device_source"] = self.audit["device_source"] or "runtime"
            self.audit["device_calls"].append(str(device))

        return super().to(*args, **kwargs)

    def forward(self,x):
        return x

    def report(self):
        actual_devices = {
                str(p.devic) for p in self.parameters()
                }

        print("=====================")
        print("===NN DEVICE AUDIT===")
        print("=====================")
        print(f"{self.audit}")
        print(f"Actual parameter devices: {actual_devices}")


