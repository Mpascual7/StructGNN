import os
print("Working directory:", os.getcwd())
import torch
from torch_geometric.data import Data
from GNN.models import Structure_GraphNetwork  # adjust if you're using a different model

# Step 1: Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: Reconstruct the model architecture (must match training!)
model = Structure_GraphNetwork(
    layer_num=3,               # Replace with the number used in training
    input_dim=11,
    hidden_dim=256,
    edge_attr_dim=3,
    aggr='mean',
    node_out_dispX_dim=1,
    node_out_dispZ_dim=1,
    node_out_momentY_dim=6,
    node_out_momentZ_dim=6,
    node_out_shearY_dim=6,
    node_out_shearZ_dim=6,
    device=device
)

# Step 3: Load the state dict (weights)
model_path = r"C:\Users\mnwpa\OneDrive\Documents\GitHub\StructGNN\Results\Static_Linear_Analysis\2025_04_17__02_26_56\model.pt"
model.load_state_dict(torch.load( model_path, map_location=device))
model.to(device)
model.eval()

# Step 4: Load the input graph data
data_test = r"C:\Users\mnwpa\OneDrive\Documents\GitHub\StructGNN\Data\Static_Linear_Analysis\structure_1\structure_graph_NodeAsNode.pt"
data = torch.load(data_test, weights_only=False)

# Move data to device
x = data.x.to(device)
edge_index = data.edge_index.to(device)
edge_attr = data.edge_attr.to(device)

# Step 5: Run inference
with torch.no_grad():
    output = model(x, edge_index, edge_attr)

# Step 6: Inspect or save output
print("Output shape:", output.shape)
print("First node prediction:", output[0])
torch.save(output, "predicted_output.pt")

# Function to extract predictions for a given node
def get_node_prediction(output: torch.Tensor, node_index: int):
    node = output[node_index]
    prediction = {
        "dispX": node[0].item(),
        "dispZ": node[1].item(),
        "momentY": node[2:8].cpu().tolist(),
        "momentZ": node[8:14].cpu().tolist(),
        "shearY": node[14:20].cpu().tolist(),
        "shearZ": node[20:26].cpu().tolist()
    }
    if node.shape[0] > 26:
        prediction["extra"] = node[26:].cpu().tolist()
    return prediction

# Example: Get prediction for node 12
node_index = 0
prediction = get_node_prediction(output, node_index)
print(f"\nFormatted prediction for node {node_index}:")
for key, value in prediction.items():
    print(f"{key}: {value}")