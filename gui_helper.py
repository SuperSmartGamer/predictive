import numpy as np
from vispy import app, scene

class GraphEngine:
    def __init__(self, title="Lightning Graph Renderer", width=1280, height=720):
        """Initializes the GPU-accelerated canvas and camera."""
        self.canvas = scene.SceneCanvas(keys='interactive', show=True, title=title, size=(width, height))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'panzoom'
        
        # GPU Visual Nodes
        self.nodes_visual = scene.visuals.Markers(parent=self.view.scene)
        
        # GPU Visual Edges (connect='segments' means every pair of coordinates is a line)
        self.edges_visual = scene.visuals.Line(parent=self.view.scene, connect='segments')
        
        # Internal state buffers
        self.positions = np.empty((0, 2), dtype=np.float32)
        self.connections = np.empty((0, 2), dtype=np.int32)
        
        # Animation variables
        self.update_callback = None
        self.timer = app.Timer('auto', connect=self._on_timer, start=False)

    def set_graph(self, positions, connections, node_colors='white', node_sizes=5, edge_colors='gray'):
        """
        The master helper function. Call this to initialize or completely rebuild the graph.
        It auto-scales to whatever amount of nodes and connections you pass.
        
        - positions: (N, 2) array of X, Y coordinates.
        - connections: (M, 2) array of integer indices (e.g., [0, 1] connects node 0 to node 1).
        """
        self.positions = np.asarray(positions, dtype=np.float32)
        self.connections = np.asarray(connections, dtype=np.int32)
        
        # Push node data to GPU
        self.nodes_visual.set_data(pos=self.positions, face_color=node_colors, size=node_sizes)
        
        # Push edge data to GPU
        self._update_edge_buffers(edge_colors)
            
        # Auto-fit the camera to your data
        self.view.camera.set_range()

    def update_positions(self, new_positions):
        """
        Helper for movement: Call this inside your update loop to move nodes instantly.
        It avoids rebuilding the connection structure, saving massive overhead.
        """
        self.positions = np.asarray(new_positions, dtype=np.float32)
        
        # Fast update node positions
        self.nodes_visual.set_data(pos=self.positions)
        
        # Fast update edge positions
        self._update_edge_buffers(color=None) # Keep existing colors

    def _update_edge_buffers(self, color=None):
        """Internal helper to slice the position array using connection indices."""
        if len(self.connections) > 0:
            # This is the magic line: it translates connection indices into X,Y pairs instantly
            edge_coords = self.positions[self.connections].reshape(-1, 2)
            if color is not None:
                self.edges_visual.set_data(pos=edge_coords, color=color)
            else:
                self.edges_visual.set_data(pos=edge_coords)
        else:
            self.edges_visual.set_data(pos=np.empty((0, 2), dtype=np.float32))

    def run(self, update_function=None):
        """Starts the renderer. Pass an update loop function for animation."""
        if update_function:
            self.update_callback = update_function
            self.timer.start()
        app.run()

    def _on_timer(self, event):
        """Fires every frame (approx 60 FPS) to call your custom logic."""
        if self.update_callback:
            self.update_callback(self)


# ==========================================
# USAGE EXAMPLE & STRESS TEST
# ==========================================
if __name__ == '__main__':
    # 1. Setup Data Scale
    NUM_NODES = 1000
    NUM_CONNECTIONS = 5000

    # 2. Generate Random Initial State
    # Positions: 1000 random (X, Y) points
    positions = np.random.normal(size=(NUM_NODES, 2), scale=100)
    
    # Connections: 5000 random pairs of indices
    connections = np.random.randint(0, NUM_NODES, size=(NUM_CONNECTIONS, 2))
    
    # Colors: Random RGBA for each node
    colors = np.random.uniform(0.5, 1.0, size=(NUM_NODES, 4))
    colors[:, 3] = 1.0 # Set alpha to fully opaque

    # 3. Initialize Engine
    engine = GraphEngine(title=f"Stress Test: {NUM_NODES} Nodes, {NUM_CONNECTIONS} Edges")
    
    # 4. Push initial data using the easy helper
    engine.set_graph(
        positions=positions, 
        connections=connections, 
        node_colors=colors, 
        node_sizes=6, 
        edge_colors=(1, 1, 1, 0.1) # Faint white lines
    )

    # 5. Define an update loop for movement
    # We will just apply a tiny bit of random "jitter" to the nodes every frame
    def my_physics_loop(eng):
        # Read current positions
        current_pos = eng.positions
        
        # Calculate new positions (Brownian motion / jitter)
        jitter = np.random.normal(scale=0.5, size=current_pos.shape)
        new_pos = current_pos + jitter
        
        # Use the fast update helper
        eng.update_positions(new_pos)

    # 6. Start the engine
    print("Starting engine. You can pan (Left Click + Drag) and zoom (Scroll).")
    engine.run(update_function=my_physics_loop)