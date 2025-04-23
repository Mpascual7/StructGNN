import torch

file_path = r'C:\Users\mnwpa\OneDrive\Documents\GitHub\StructGNN\Data\Static_Linear_Analysis\structure_1\structure_graph_NodeAsNode.pt'

# Attempt to load the file with weights_only=False to bypass the error
structure_graph = torch.load(file_path, weights_only=False)

# Continue with your code, for example:
print(structure_graph)
