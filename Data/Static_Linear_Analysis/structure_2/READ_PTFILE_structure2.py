import torch

file_path = r'C:\Users\mnwpa\OneDrive\Documents\GitHub\StructGNN\Data\Static_Linear_Analysis\structure_2\structure_graph_NodeAsNode.pt'

# Attempt to load the file with weights_only=False to bypass the error
structure_graph = torch.load(file_path, weights_only=False)

# Continue with your code, for example:
print(structure_graph)
print("Node features:\n", structure_graph.x)
print("Edge index:\n", structure_graph.edge_index)
print("Number of nodes:", structure_graph.num_nodes)
print("Number of edges:", structure_graph.num_edges)
print("Is directed:", structure_graph.is_directed())
print("Contains isolated nodes:", structure_graph.has_isolated_nodes())
print("Contains self-loops:", structure_graph.has_self_loops())
