import torch

file_path = r"C:\Users\mnwpa\OneDrive\Documents\GitHub\StructGNN\Results\Static_Linear_Analysis\2025_04_15__02_51_59\model.pt"
# Attempt to load the file with weights_only=False to bypass the error
structure_graph = torch.load(file_path, weights_only=False)

# Continue with your code, for example:
print(structure_graph)