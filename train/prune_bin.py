import sys
import torch

state_dict = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
new_state_dict = {}
for k, v in state_dict.items():
    assert k.startswith("model."), k
    new_state_dict[k[6:]] = v
torch.save(new_state_dict, sys.argv[1])