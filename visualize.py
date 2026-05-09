import torch
import json
import numpy as np

print("Loading telemetry data...")

# FIX: Added weights_only=False to bypass the PyTorch 2.6 security check
try:
    log = torch.load('graph_telemetry.pt', weights_only=False)
except Exception as e:
    print(f"Failed to load: {e}")
    exit()

json_log = []

print(f"Processing {len(log)} snapshots...")

for snapshot in log:
    # We need to ensure everything is a standard Python list for JSON compatibility
    json_log.append({
        'epoch': int(snapshot['epoch']),
        'batch': int(snapshot['batch']),
        'accuracy': float(snapshot['accuracy']),
        'coords': snapshot['coords'].tolist(),     # Convert numpy to list
        'adjacency': snapshot['adjacency'].tolist() # Convert numpy to list
    })

print("Saving to graph_telemetry.json...")
with open('graph_telemetry.json', 'w') as f:
    json.dump(json_log, f)

print("Done! You can now drop graph_telemetry.json into the visualizer.")