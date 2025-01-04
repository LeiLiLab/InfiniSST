import sys
import torch

print("Loading model from", sys.argv[1])
state_dict = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
new_state_dict = {}
for k, v in state_dict.items():
    assert k.startswith("model."), k
    new_state_dict[k[6:]] = v
print("Saving model to", sys.argv[1])
torch.save(new_state_dict, sys.argv[1])