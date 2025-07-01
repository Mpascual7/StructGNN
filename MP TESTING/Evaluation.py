import os
import sys
from datetime import datetime

# Add parent directory to sys.path before importing GNN
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from torch_geometric.data import Data
from GNN.models import Structure_GraphNetwork

# Print working directory
print("Working directory:", os.getcwd())

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the folder containing trained models
checkpoints_root = os.path.normpath(os.path.join(script_dir, '..', 'Results', 'Static_Linear_Analysis'))

# === Find the latest checkpoint folder dynamically BEFORE creating new output folder ===
folders = sorted(
    [f for f in os.listdir(checkpoints_root)
     if os.path.isdir(os.path.join(checkpoints_root, f)) and f.startswith('20')],
    reverse=True
)

if not folders:
    raise FileNotFoundError("No checkpoint folders found in Static_Linear_Analysis.")

latest_folder = folders[0]
print(f"Loading model from latest checkpoint folder: {latest_folder}")

pretrained_model_path = os.path.join(checkpoints_root, latest_folder, 'model.pt')
if not os.path.exists(pretrained_model_path):
    raise FileNotFoundError(f"model.pt not found in: {pretrained_model_path}")

# === Setup output directory inside Predicted_Outputs with timestamp ===
predicted_outputs_root = os.path.normpath(os.path.join(script_dir, '..', 'Results', 'Predicted_Outputs'))
os.makedirs(predicted_outputs_root, exist_ok=True)

timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
output_path = os.path.join(predicted_outputs_root, timestamp)
os.makedirs(output_path, exist_ok=True)
print(f"Saving predicted output to new folder: {output_path}")

# Reconstruct the model architecture (must match training!)
model = Structure_GraphNetwork(
    layer_num=3,
    input_dim=11,
    hidden_dim=512,
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

model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
model.to(device)
model.eval()

# === Load input graph data ===
data_relative_path = os.path.join('Data', 'Static_Linear_Analysis', 'structure_1', 'structure_graph_NodeAsNode_pseudo.pt')
data_path = os.path.normpath(os.path.join(script_dir, '..', data_relative_path))
print(f"Loading graph data from: {data_path}")
data = torch.load(data_path, weights_only=False)

# Move data to device
x = data.x.to(device)
edge_index = data.edge_index.to(device)
edge_attr = data.edge_attr.to(device)

# === Run inference ===
with torch.no_grad():
    output = model(x, edge_index, edge_attr)

# Inspect and save output
print("Output shape:", output.shape)
print("First node prediction:", output[0])

# Save predicted output to the new folder
output_file = os.path.join(output_path, "predicted_output.pt")
torch.save(output, output_file)
print(f"Saved predicted output to: {output_file}")

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

# Fancy formatted output for node prediction
node_index = 0
prediction = get_node_prediction(output, node_index)

print(f"\nModel prediction for node {node_index}:\n" + "-"*40)
for key, value in prediction.items():
    if isinstance(value, list):
        print(f"{key:<10}: ", end="")
        formatted_list = ", ".join(f"{v:8.4f}" for v in value)
        print(formatted_list)
    else:
        print(f"{key:<10}: {value:.6f}")
print("-"*40)