import torch
import json
import os
import shutil
from torchvision import datasets, transforms
from collections import deque
import time
import random
from model2 import SpatialDynamicGraph

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INIT] Device: {device}")
    graph = SpatialDynamicGraph(num_nodes=150, state_dim=16).to(device)

    transform = transforms.Compose([
        transforms.Resize((7, 7)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    print("[INIT] Downloading and compressing Full MNIST Dataset...")
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    all_imgs_list = []
    all_labels_list = []
    
    for img, lbl in dataset:
        all_imgs_list.append(img)
        all_labels_list.append(lbl)
        
    full_imgs = torch.stack(all_imgs_list).to(device)
    full_labels = torch.tensor(all_labels_list, device=device)

    VAL_SIZE = 100 
    val_imgs = full_imgs[:VAL_SIZE]
    val_labels = full_labels[:VAL_SIZE]
    train_imgs = full_imgs[VAL_SIZE:]
    train_labels = full_labels[VAL_SIZE:]

    print(f"[INIT] Continuous Validation Set: {len(val_imgs)} Images")
    print(f"[INIT] Infinite Training Pool: {len(train_imgs)} Images")

    comp_val_pixels  = graph.retina_compressor(val_imgs)
    val_retina_states = torch.tanh(comp_val_pixels.view(VAL_SIZE, 10, 16))

    if os.path.exists('telemetry'): shutil.rmtree('telemetry')
    os.makedirs('telemetry', exist_ok=True)
    meta_index  = []
    chunk_frames = []

    rolling_correct = deque(maxlen=200)
    class_correct = {i: 0 for i in range(10)}
    class_total   = {i: 0 for i in range(10)}

    print(f"=== FULL MNIST GENERALIZATION ENGINE ===")

    def get_random_stimulus():
        idx = random.randint(0, len(train_imgs) - 1)
        return {'id': idx, 'img': train_imgs[idx], 'label': int(train_labels[idx].item())}

    current_stim = get_random_stimulus()
    val_acc = 0.0
    best_val_acc = 0.0
    stability_counter = 0
    STABILITY_THRESHOLD = 200
    swap_count = 0
    predicted_label = -1

    t_start = time.time()
    MAX_STEPS = 1_000_000 
    
    CHUNK_SIZE = 2 
    
    for step in range(MAX_STEPS):
        comp_pixels = graph.retina_compressor(current_stim['img'].unsqueeze(0))
        retina_states = torch.tanh(comp_pixels.view(1, 10, 16))

        # 1. THE TRULY HONEST PREDICTION PEEK (Unclamped)
        predicted_label = -1
        is_correct = False
        if graph.current_states is not None:
            peek_states = graph.current_states.clone()
            
            # FIX: Scrub the motor nodes so it can't cheat by reading the previous step's clamped truth
            peek_states[:, graph.motor_idx, :] = torch.randn(1, 10, graph.state_dim, device=device) * 0.1

            for _ in range(5): 
                peek_states = graph.inference_step(graph.retina_idx, retina_states, peek_states)
            
            with torch.no_grad():
                motor_logits = peek_states[:, graph.motor_idx, :].mean(dim=2)
                predicted_label = int(torch.argmax(motor_logits, dim=1).item())
                true_label = int(current_stim['label'])
                is_correct = (predicted_label == true_label)
                
                rolling_correct.append(float(is_correct))
                class_total[true_label]   += 1
                class_correct[true_label] += int(is_correct)

        rolling_acc = 100.0 * sum(rolling_correct) / max(len(rolling_correct), 1)

        # 2. CLAMP AND TRAIN (Apply truth force)
        motor_targets = torch.full((1, 10, 16), -0.9, device=device)
        motor_targets[0, current_stim['label'], :] = 0.9

        all_idx = graph.retina_idx + graph.motor_idx
        all_vals = torch.cat([retina_states, motor_targets], dim=1)

        _, _, _ = graph.continuous_step(all_idx, all_vals)
        loss, surprises, raw_surprises = graph.train_step_pure_pc(surprise_threshold=0.1)

        if step % 50 == 0:
            graph.physics_step(decay_rate=0.005)

        # 3. FULL BATCH VALIDATION PROBE 
        if step % 50 == 0: # Reduced frequency to save computation
            val_states = torch.rand(VAL_SIZE, graph.num_nodes, graph.state_dim, device=device) * 0.1
            for _ in range(5): 
                val_states = graph.inference_step(graph.retina_idx, val_retina_states, val_states)

            with torch.no_grad():
                logits = val_states[:, graph.motor_idx, :].mean(dim=2)
                _, preds = torch.max(logits, 1)
                val_acc = float((preds == val_labels).float().mean().item() * 100.0)
                
                # FIX: Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(graph.state_dict(), 'best_model.pth')
                    print(f"   [SAVED] New best model at {val_acc:.1f}% accuracy.")

        # 4. HOMEOSTASIS SWAP 
        if loss < 0.05 and val_acc > 70.0:
            stability_counter += 1
        else:
            stability_counter = 0

        if stability_counter >= STABILITY_THRESHOLD:
            swap_count += 1
            old_label = current_stim['label']
            current_stim = get_random_stimulus()
            print(f"[SWAP  {step:06d}] Stable #{swap_count:03d} | {old_label}→{current_stim['label']} | V-Acc: {val_acc:.1f}%")
            stability_counter = 0

        # 5. CONSOLE LOGGING
        if step % 100 == 0 or step < 10:
            elapsed = time.time() - t_start
            syn_count = int(graph.adjacency.sum().item())
            if raw_surprises is not None:
                raw_np = raw_surprises.numpy()
                worst_node  = int(raw_np.argmax())
                worst_surp  = float(raw_np.max())
                node_type   = ("RETINA" if worst_node < 10 else "MOTOR" if worst_node < 20 else "HIDDEN")
            else:
                worst_node, worst_surp, node_type = -1, 0.0, "?"

            print(
                f"[{step:06d}] {elapsed:6.0f}s | Tgt:{current_stim['label']} Pred:{predicted_label} {'✓' if is_correct else '✗'} | "
                f"Roll:{rolling_acc:5.1f}% Val:{val_acc:5.1f}% | Loss:{loss:.4f} | Syns:{syn_count} | WorstNode:{worst_node}({node_type})"
            )

        # 6. TELEMETRY PACKAGING (Optimized list generation)
        s_surp = torch.round(torch.nan_to_num(surprises, nan=0.0), decimals=3).tolist() if surprises is not None else [0.0]*graph.num_nodes
        s_raw  = torch.round(torch.nan_to_num(raw_surprises, nan=0.0), decimals=3).tolist() if raw_surprises is not None else [0.0]*graph.num_nodes
        s_coor = torch.round(torch.nan_to_num(graph.coords.detach().cpu(), nan=0.0), decimals=3).tolist()

        chunk_frames.append({
            'step':          int(step),
            'rolling_acc':   float(rolling_acc),
            'val_acc':       float(val_acc),
            'loss':          float(torch.nan_to_num(torch.tensor(loss), nan=0.0).item()),
            'true_label':    int(current_stim['label']),
            'pred_label':    int(predicted_label),
            'correct':       bool(is_correct),
            'swap_count':    int(swap_count),
            'node_surprises': s_surp,
            'raw_surprises':  s_raw,
            'coords':         s_coor,
            'adjacency':      graph.adjacency.detach().cpu().numpy().tolist(),
            'syn_count':      int(graph.adjacency.sum().item()),
        })

        if (step + 1) % CHUNK_SIZE == 0:
            chunk_id = step // CHUNK_SIZE
            filename = f"chunk_{chunk_id:04d}.json"
            with open(os.path.join('telemetry', filename), 'w') as f:
                json.dump(chunk_frames, f)
            meta_index.append(filename)
            with open(os.path.join('telemetry', 'meta.json.tmp'), 'w') as f:
                json.dump(meta_index, f)
            
            max_retries = 10
            for i in range(max_retries):
                try:
                    os.replace(os.path.join('telemetry', 'meta.json.tmp'), os.path.join('telemetry', 'meta.json'))
                    break
                except PermissionError as e:
                    if i == max_retries - 1:
                        print(f"[ERROR] Failed to update meta.json after {max_retries} attempts: {e}")
                        raise
                    time.sleep(0.1 * (i + 1)) 
            
            chunk_frames = []

if __name__ == "__main__":
    main()