import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_structure(structure_id):
    """Load PyTorch Geometric data for specified structure"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root (StructGNN directory)
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(
        project_root,
        "Data",
        "Static_Linear_Analysis",
        f"structure_{structure_id}",
        "structure_graph_NodeAsNode.pt"
    )
    
    # Validate path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No data found for structure {structure_id} at: {data_path}")
    
    print(f"Loading structure {structure_id} from:\n{os.path.normpath(data_path)}")
    return torch.load(data_path, weights_only=False)

def visualize_graph(structure_graph, structure_id):
    """Generate 3D visualization of structure graph"""
    # Extract node positions [x, y, z]
    positions = {
        i: (structure_graph.x[i, 3].item(), 
            structure_graph.x[i, 4].item(), 
            structure_graph.x[i, 5].item())
        for i in range(structure_graph.num_nodes)
    }
    
    # Build networkx graph
    G = nx.Graph()
    edge_index = structure_graph.edge_index.numpy()
    for src, tgt in edge_index.T:
        G.add_edge(src, tgt)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot edges
    for edge in G.edges():
        x = [positions[edge[0]][0], positions[edge[1]][0]]
        y = [positions[edge[0]][1], positions[edge[1]][1]]
        z = [positions[edge[0]][2], positions[edge[1]][2]]
        ax.plot(x, y, z, c='b', lw=0.8)
    
    # Plot nodes
    xs, ys, zs = zip(*[positions[i] for i in G.nodes])
    ax.scatter(xs, ys, zs, c='r', s=15)
    
    # Add node labels
    for node_id, (x, y, z) in positions.items():
        ax.text(x, y, z + 0.1, str(node_id), fontsize=7, ha='center')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Structural Graph (Structure {structure_id})', pad=20)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Get user input with validation
    while True:
        try:
            structure_id = int(input("Enter structure number (1-2000): "))
            if 1 <= structure_id <= 2000:
                break
            print("Error: Number must be between 1-2000")
        except ValueError:
            print("Error: Please enter a valid integer")
    
    # Load and visualize
    structure_graph = load_structure(structure_id)
    
    # Print graph info
    print("\nGraph Metadata:")
    print(f"Nodes: {structure_graph.num_nodes} | Edges: {structure_graph.num_edges}")
    print(f"Directed: {structure_graph.is_directed()}")
    print(f"Isolated nodes: {structure_graph.has_isolated_nodes()}")
    print(f"Self-loops: {structure_graph.has_self_loops()}")
    
    visualize_graph(structure_graph, structure_id)
