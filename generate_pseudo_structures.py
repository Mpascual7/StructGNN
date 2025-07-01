import os
import torch
import numpy as np
from torch_geometric.data import Data


def generate_structure_pseudo(structure_id, output_dir="Data_Generated"):
    # Randomize parameters with caps
    grid_x = np.random.randint(2, 7)  # 2-6 nodes in x-direction
    grid_z = np.random.randint(2, 7)  # 2-6 nodes in z-direction (depth)
    num_floors = np.random.randint(2, 7)  # 1-6 floors (Y-direction)
    
    # Create nodes with proper numbering: Y -> Z -> X (floor by floor, row by row)
    nodes = []
    node_id = 0
    
    # First create the regular structure nodes
    for y in range(num_floors):       # Y position (floor level) - FIRST
        for z in range(grid_z):       # Z position (depth/rows) - SECOND
            for x in range(grid_x):   # X position (left to right) - LAST
                # Feature structure: [span, story, ID, x, y, z, DoF, mass_flag, load, angle, dist]
                mass_flag = 1.0 if (x == 0 and z == 0) or (x == grid_x-1 and z == 0) else 0.0
                load = np.random.uniform(0.003, 0.015)
                
                nodes.append([
                    grid_x,              # Span index (x-nodes)
                    grid_z,              # Story index (z-nodes)
                    num_floors,          # Total floors
                    float(x),            # X position
                    float(y),            # Y position (vertical - floor level)
                    float(z),            # Z position (depth)
                    0.0 if (x == 0 and z == 0) else 1.0,  # DoF (fixed at origin)
                    mass_flag,           # Mass flag
                    load,                # Structural load
                    0.0,                 # Angle
                    0.0                  # Distance
                ])
                node_id += 1
    
    # Then create the pseudo pillar nodes at (-1, -1, -1) for each floor level (LAST)
    for y in range(num_floors):
        # Pseudo node for this floor level
        nodes.append([
            grid_x,              # Span index (x-nodes)
            grid_z,              # Story index (z-nodes)
            num_floors,          # Total floors
            -1.0,                # X position (pseudo pillar)
            float(y),            # Y position (vertical - floor level)
            -1.0,                # Z position (pseudo pillar)
            0.0,                 # DoF (fixed for pseudo nodes)
            0.0,                 # Mass flag (no mass for pseudo nodes)
            0.0,                 # Structural load (no load for pseudo nodes)
            0.0,                 # Angle
            0.0                  # Distance
        ])
        node_id += 1

    # Create edges
    edges = []
    
    # Helper function to get node index based on x, z, y coordinates
    # Regular nodes start at index 0
    # Pseudo nodes are at the end: starting from total_regular_nodes
    def get_node_index(x, z, y):
        return y * (grid_x * grid_z) + z * grid_x + x
    
    def get_pseudo_node_index(y):
        total_regular_nodes = num_floors * grid_x * grid_z
        return total_regular_nodes + y
    
    # Horizontal connections (within floors) - SKIP bottom floor for table effect
    for y in range(num_floors):
        # Skip horizontal connections for bottom floor (y=0) to create table legs
        if y == 0:
            continue
            
        for x in range(grid_x):
            for z in range(grid_z):
                current = get_node_index(x, z, y)
                
                # Right neighbor (X direction)
                if x < grid_x - 1:
                    right = get_node_index(x + 1, z, y)
                    edges.append([current, right])
                    edges.append([right, current])  # Undirected
                
                # Depth neighbor (Z direction)
                if z < grid_z - 1:
                    depth = get_node_index(x, z + 1, y)
                    edges.append([current, depth])
                    edges.append([depth, current])  # Undirected
    
    # Vertical connections (between floors) - Y direction
    for x in range(grid_x):
        for z in range(grid_z):
            for y in range(num_floors - 1):
                current = get_node_index(x, z, y)
                above = get_node_index(x, z, y + 1)
                edges.append([current, above])
                edges.append([above, current])  # Undirected
    
    # Vertical connections between pseudo nodes (pseudo pillar)
    for y in range(num_floors - 1):
        current_pseudo = get_pseudo_node_index(y)
        above_pseudo = get_pseudo_node_index(y + 1)
        edges.append([current_pseudo, above_pseudo])
        edges.append([above_pseudo, current_pseudo])  # Undirected
    
    # Connections from each floor to its corresponding pseudo node
    for y in range(num_floors):
        pseudo_node = get_pseudo_node_index(y)
        
        # Connect all nodes on this floor to the pseudo node
        for x in range(grid_x):
            for z in range(grid_z):
                regular_node = get_node_index(x, z, y)
                edges.append([pseudo_node, regular_node])
                edges.append([regular_node, pseudo_node])  # Undirected

    # Convert to tensors
    node_features = torch.tensor(nodes, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Create graph data object
    structure_graph = Data(
        x=node_features,
        edge_index=edge_index,
        y=torch.randn(node_features.shape[0], 38)  # Random targets
    )
    
    # Save structure
    struct_dir = os.path.join(output_dir, f"structure_{structure_id}")
    os.makedirs(struct_dir, exist_ok=True)
    save_path = os.path.join(struct_dir, "structure_graph_NodeAsNode_pseudo.pt")
    torch.save(structure_graph, save_path)
    
    return structure_graph, save_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate random building structures with pseudo nodes")
    parser.add_argument("--start_id", type=int, required=True, help="Starting structure ID")
    parser.add_argument("--num_structures", type=int, default=1, help="Number to generate")
    args = parser.parse_args()
    
    for i in range(args.num_structures):
        structure_id = args.start_id + i
        structure, path = generate_structure_pseudo(structure_id)
        print(f"Generated pseudo structure {structure_id}: "
              f"{structure.num_nodes} nodes, "
              f"{structure.num_edges} edges → {path}")
