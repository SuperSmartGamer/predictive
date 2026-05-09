import torch
import json
import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from model import SpatialDynamicGraph

def compress_list(data, precision=3):
    if isinstance(data, list): return [compress_list(x, precision) for x in data]
    elif isinstance(data, float): return round(data, precision)
    return data

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = SpatialDynamicGraph(num_nodes=75, state_dim=16).to(device)
    transform = transforms.Compose([transforms.Resize((7, 7)), transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)

    # THE FIX: Setup Sharded Telemetry Directory
    if os.path.exists('telemetry'):
        shutil.rmtree('telemetry') # Clear old runs
    os.makedirs('telemetry', exist_ok=True)
    meta_index = []

    print("Engine Running: SHARDED TELEMETRY ACTIVE. Memory safe.")
    
    for epoch in range(15):
        correct = total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)

            retina_states = torch.tanh(graph.retina_compressor(images).view(batch_size, 10, 16))
            motor_targets = torch.full((batch_size, 10, 16), -0.9, device=device)
            for b in range(batch_size): motor_targets[b, labels[b], :] = 0.9

            do_capture = True # We can leave this True now!

            free_states, _ = graph.relax(graph.retina_idx, retina_states, max_steps=50, capture_history=False)
            graph.free_states = free_states.clone()
            
            with torch.no_grad():
                logits = graph.free_states[:, graph.motor_idx, :].mean(dim=2)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0); correct += (predicted == labels).sum().item()

            clamped_states, p3_history = graph.relax(
                graph.retina_idx + graph.motor_idx, 
                torch.cat([retina_states, motor_targets], dim=1), 
                max_steps=50, 
                init_states=graph.free_states, 
                capture_history=do_capture
            )
            graph.clamped_states = clamped_states.clone()

            pc_loss, node_surprises = graph.train_step_pure_pc(surprise_threshold=0.5)
            graph.physics_step()

            # --- SHARDED LOGGING ---
            batch_frames = []
            
            if do_capture:
                for h in p3_history:
                    s_surp = torch.nan_to_num(torch.tensor(h['surprises']), nan=0.0).tolist()
                    s_coor = torch.nan_to_num(torch.tensor(h['coords']), nan=0.0).tolist()
                    batch_frames.append({
                        'epoch': epoch + 1, 'batch': batch_idx, 'substep': h['step'],
                        'accuracy': round(100 * correct / total, 1), 
                        'node_surprises': compress_list(s_surp, 3),
                        'coords': compress_list(s_coor, 3), 
                        'adjacency': graph.adjacency.detach().cpu().numpy().tolist()
                    })
            
            # 1. Write the specific batch to its own file
            filename = f"batch_{epoch+1}_{batch_idx:03d}.json"
            filepath = os.path.join('telemetry', filename)
            with open(filepath, 'w') as f:
                json.dump(batch_frames, f)
                
            # 2. Update the Meta Index
            meta_index.append(filename)
            with open(os.path.join('telemetry', 'meta.json.tmp'), 'w') as f:
                json.dump(meta_index, f)
            
            # Atomic replace the meta file (the only file the browser polls)
            for _ in range(10):
                try:
                    os.replace(os.path.join('telemetry', 'meta.json.tmp'), os.path.join('telemetry', 'meta.json'))
                    break
                except PermissionError:
                    time.sleep(0.05)

            if batch_idx % 10 == 0: 
                print(f"E{epoch+1} B{batch_idx} | Acc: {100*correct/total:.2f}% | Syn: {graph.adjacency.sum().item()}")

if __name__ == "__main__":
    main()