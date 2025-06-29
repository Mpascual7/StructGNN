import os
import torch

def load_structure(structure_id):
    """Load PyTorch Geometric data for specified structure"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the project root (StructGNN directory)
    project_root = os.path.dirname(script_dir)
    
    data_path = os.path.join(
        project_root,  # Now points to ~/icode/StructGNN
        "Data",
        "Static_Linear_Analysis",
        f"structure_{structure_id}",
        "structure_graph_NodeAsNode.pt"
    )
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No data found for structure {structure_id} at: {data_path}")
    
    print(f"Loading structure {structure_id} from:\n{os.path.normpath(data_path)}")
    return torch.load(data_path, weights_only=False)

def print_diagnostics(structure_graph):
    """Print comprehensive graph diagnostics"""
    print("\n" + "="*50)
    print(f"{'STRUCTURE DIAGNOSTICS':^50}")
    print("="*50)
    
    # Basic graph properties
    print(f"\n{'[GRAPH PROPERTIES]':^50}")
    print(f"Number of nodes: {structure_graph.num_nodes}")
    print(f"Number of edges: {structure_graph.num_edges}")
    print(f"Is directed: {structure_graph.is_directed()}")
    print(f"Contains isolated nodes: {structure_graph.has_isolated_nodes()}")
    print(f"Contains self-loops: {structure_graph.has_self_loops()}")
    
    # Node features
    print(f"\n{'[NODE FEATURES]':^50}")
    print(f"Feature tensor shape: {structure_graph.x.shape}")
    print("All node features:")
    for i in range(structure_graph.num_nodes):
        print(f"Node {i}: {structure_graph.x[i].tolist()}")

    #Edge features
    print("All edge features:")
    for i in range(structure_graph.edge_attr.size(0)):
        print(f"Edge {i}: {structure_graph.edge_attr[i].tolist()}")
    
    # Edge information
    print(f"\n{'[EDGE INFORMATION]':^50}")
    print(f"Edge index shape: {structure_graph.edge_index.shape}")
    print("All edges:")
    for i in range(structure_graph.edge_index.shape[1]):
        src, tgt = structure_graph.edge_index[:, i].tolist()
        print(f"Edge {i}: {src} → {tgt}")
    
    # Additional diagnostics
    print(f"\n{'[ADDITIONAL PROPERTIES]':^50}")
    print(f"Graph keys: {list(structure_graph.keys())}")
    if 'y' in structure_graph:
        print(f"Target values: {structure_graph.y.shape}")

if __name__ == "__main__":
    while True:
        try:
            structure_id = int(input("Enter structure number (1-2000): "))
            if 1 <= structure_id <= 2000:
                break
            print("Error: Number must be between 1-2000")
        except ValueError:
            print("Error: Please enter a valid integer")
    
    structure_graph = load_structure(structure_id)
    print_diagnostics(structure_graph)
