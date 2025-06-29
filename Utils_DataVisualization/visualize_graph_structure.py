import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk, messagebox

def load_structure(structure_id, is_generated=False, is_pseudo=False):
    """Load PyTorch Geometric data for specified structure"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root (StructGNN directory)
    project_root = os.path.dirname(script_dir)
    
    # Determine the data directory based on user choice
    if is_generated:
        data_dir = "Data_Generated"
    else:
        data_dir = os.path.join("Data", "Static_Linear_Analysis")
    
    # Determine filename based on structure type
    if is_pseudo:
        filename = "structure_graph_NodeAsNode_pseudo.pt"
    else:
        filename = "structure_graph_NodeAsNode.pt"
    
    data_path = os.path.join(
        project_root,
        data_dir,
        f"structure_{structure_id}",
        filename
    )
    
    # If pseudo file doesn't exist but we're looking for generated structures,
    # try the regular filename as fallback
    if is_generated and not os.path.exists(data_path):
        fallback_filename = "structure_graph_NodeAsNode.pt"
        fallback_path = os.path.join(
            project_root,
            data_dir,
            f"structure_{structure_id}",
            fallback_filename
        )
        if os.path.exists(fallback_path):
            data_path = fallback_path
            print(f"Pseudo file not found, using regular format: {fallback_filename}")
    
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

class StructureVisualizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Structure Visualizer")
        self.root.geometry("350x280")
        self.root.resizable(False, False)
        
        # Apply modern theme
        self.set_modern_theme()
        
        # Create variables
        self.source_var = tk.StringVar(value="base")
        self.structure_id_var = tk.StringVar()
        self.pseudo_var = tk.BooleanVar(value=False)
        
        # Create GUI elements
        self.create_widgets()
        
        # Set focus to entry field
        self.id_entry.focus_set()
        
    def set_modern_theme(self):
        """Apply modern theme with accent color"""
        style = ttk.Style()
        
        # Try to use system-specific theme first
        for theme in ['vista', 'xpnative', 'aqua', 'clam']:
            if theme in style.theme_names():
                style.theme_use(theme)
                break
        
        # Configure colors
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Segoe UI', 9))
        style.configure('TRadiobutton', background='#f0f0f0', font=('Segoe UI', 9))
        style.configure('TCheckbutton', background='#f0f0f0', font=('Segoe UI', 9))
        style.configure('TButton', font=('Segoe UI', 9), padding=5)
        style.configure('TLabelframe', background='#f0f0f0', font=('Segoe UI', 9, 'bold'))
        style.configure('TLabelframe.Label', background='#f0f0f0')
        
        # Configure button colors
        style.map('TButton',
            background=[('active', '#e0e0e0'), ('pressed', '#d0d0d0')],
            foreground=[('active', 'black')]
        )
        
        # Set window background
        self.root.configure(background='#f0f0f0')
        
    def create_widgets(self):
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding=(15, 15))
        main_frame.pack(fill="both", expand=True)
        
        # Source selection frame
        source_frame = ttk.LabelFrame(main_frame, text="Data Source")
        source_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Radiobutton(
            source_frame, text="Base Structures", 
            variable=self.source_var, value="base",
            command=self.on_source_change
        ).pack(anchor="w", padx=10, pady=5)
        
        ttk.Radiobutton(
            source_frame, text="Generated Structures", 
            variable=self.source_var, value="generated",
            command=self.on_source_change
        ).pack(anchor="w", padx=10, pady=5)
        
        # Pseudo structure option (only for generated)
        self.pseudo_check = ttk.Checkbutton(
            source_frame, text="Use Pseudo Structure Format", 
            variable=self.pseudo_var
        )
        self.pseudo_check.pack(anchor="w", padx=30, pady=2)
        
        # Structure ID entry
        id_frame = ttk.Frame(main_frame)
        id_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(id_frame, text="Structure ID:").pack(side="left", padx=(0, 10))
        
        self.id_entry = ttk.Entry(id_frame, textvariable=self.structure_id_var, width=10)
        self.id_entry.pack(side="left", fill="x", expand=True)
        
        # Bind Enter key to visualize
        self.id_entry.bind("<Return>", lambda event: self.on_visualize())
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x")
        
        # Configure buttons with styling
        ttk.Button(
            button_frame, text="Cancel", 
            command=self.root.destroy,
            style="TButton"
        ).pack(side="right", padx=(5, 0))
        
        self.visualize_btn = ttk.Button(
            button_frame, text="Visualize", 
            command=self.on_visualize,
            style="Accent.TButton"
        )
        self.visualize_btn.pack(side="right")
        
        # Create custom style for the primary button
        style = ttk.Style()
        style.configure('Accent.TButton', background='#4a6baf', foreground='white')
        style.map('Accent.TButton',
            background=[('active', '#3a5a9f'), ('pressed', '#2a4a8f')],
            foreground=[('active', 'white'), ('pressed', 'white')]
        )
        
        # Initialize pseudo checkbox state
        self.on_source_change()
        
    def on_source_change(self):
        """Handle source selection change"""
        if self.source_var.get() == "generated":
            self.pseudo_check.configure(state="normal")
        else:
            self.pseudo_check.configure(state="disabled")
            self.pseudo_var.set(False)
        
    def on_visualize(self):
        """Handle visualize button click"""
        # Get values from GUI
        source = self.source_var.get()
        structure_id_str = self.structure_id_var.get()
        is_pseudo = self.pseudo_var.get()
        
        # Validate inputs
        if not structure_id_str.isdigit():
            messagebox.showerror("Input Error", "Structure ID must be a positive integer")
            return
            
        structure_id = int(structure_id_str)
        if structure_id <= 0:
            messagebox.showerror("Input Error", "Structure ID must be positive")
            return
            
        # Close GUI window
        self.root.destroy()
        
        # Determine if we're loading generated structure
        is_generated = (source == "generated")
        
        try:
            # Load and visualize
            structure_graph = load_structure(structure_id, is_generated, is_pseudo)
            
            # Print graph info to console
            print("\nGraph Metadata:")
            print(f"Nodes: {structure_graph.num_nodes} | Edges: {structure_graph.num_edges}")
            print(f"Directed: {structure_graph.is_directed()}")
            print(f"Isolated nodes: {structure_graph.has_isolated_nodes()}")
            print(f"Self-loops: {structure_graph.has_self_loops()}")
            
            # Show visualization
            visualize_graph(structure_graph, structure_id)
            
        except FileNotFoundError as e:
            messagebox.showerror("File Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Create GUI
    root = tk.Tk()
    app = StructureVisualizerGUI(root)
    root.mainloop()
