import torch
import torch.nn as nn
import math

class SpatialDynamicGraph(nn.Module):
    def __init__(self, num_nodes=75, state_dim=16, space_dim=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.state_dim = state_dim
        self.space_dim = space_dim

        self.retina_compressor = nn.Linear(49, 10 * state_dim)

        hidden_dim = 30
        mlp_input_dim = state_dim + space_dim 
        
        self.W1 = nn.Parameter(torch.randn(num_nodes, mlp_input_dim, hidden_dim) / math.sqrt(mlp_input_dim))
        self.b1 = nn.Parameter(torch.zeros(num_nodes, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(num_nodes, hidden_dim, state_dim) / math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(num_nodes, state_dim))

        # Start hidden nodes with a bit more spread so they aren't a single point
        self.register_buffer('coords', torch.randn(num_nodes, space_dim) * 1.5)
        self.register_buffer('weights', torch.zeros(num_nodes, num_nodes))
        self.register_buffer('freqs', torch.zeros(num_nodes, num_nodes))
        self.register_buffer('adjacency', torch.zeros(num_nodes, num_nodes, dtype=torch.bool))

        self.retina_idx = list(range(0, 10))  
        self.motor_idx  = list(range(10, 20)) 
        self.hidden_idx = list(range(20, 75)) 

        # Anchors at +/- 4.0
        retina = [[3.0 * math.cos(2*math.pi*i/10), 3.0 * math.sin(2*math.pi*i/10), -4.0] for i in range(10)]
        motor  = [[3.0 * math.cos(2*math.pi*i/10), 3.0 * math.sin(2*math.pi*i/10), 4.0] for i in range(10)]
        self.register_buffer('retina_anchors', torch.tensor(retina, dtype=torch.float32))
        self.register_buffer('motor_anchors',  torch.tensor(motor,  dtype=torch.float32))

        with torch.no_grad():
            self.coords[self.retina_idx] = self.retina_anchors
            self.coords[self.motor_idx]  = self.motor_anchors

        self._init_random_topology(density=0.20) 
        self.mlp_optimizer = torch.optim.Adam(self.parameters(), lr=0.002)

        self.free_states    = None
        self.clamped_states = None
        self.current_states = None 

    def _init_random_topology(self, density):
        rand_mask = torch.rand(self.num_nodes, self.num_nodes) < density
        rand_mask.fill_diagonal_(False)
        for r in self.retina_idx:
            for h in self.hidden_idx:
                if torch.rand(1).item() < 0.3: rand_mask[r, h] = True
        for h in self.hidden_idx:
            for m in self.motor_idx:
                if torch.rand(1).item() < 0.3: rand_mask[h, m] = True
        self.adjacency = rand_mask
        self.weights[self.adjacency] = 0.5
        self.freqs[self.adjacency]   = 1.0

    def mlp_forward(self, x):
        h1 = torch.einsum('bnmd,ndh->bnmh', x, self.W1) + self.b1.unsqueeze(1)
        h1 = torch.nn.functional.leaky_relu(h1)
        h2 = torch.einsum('bnmh,nhd->bnmd', h1, self.W2) + self.b2.unsqueeze(1)
        return torch.tanh(h2) 

    def compute_masked_energy(self, states):
        batch_size = states.shape[0]
        active_weights = self.weights * self.adjacency.float()
        total_routed = torch.einsum('nm,bmd->bnd', active_weights, states)
        direct_messages = active_weights.unsqueeze(0).unsqueeze(3) * states.unsqueeze(1)
        masked_context = total_routed.unsqueeze(2) - direct_messages 
        
        rel_pos = (self.coords.unsqueeze(0) - self.coords.unsqueeze(1)).detach() 
        rel_pos_batch = rel_pos.unsqueeze(0).expand(batch_size, -1, -1, -1)      
        
        mlp_input = torch.cat([masked_context, rel_pos_batch], dim=-1)           
        predictions = self.mlp_forward(mlp_input)
        target_states = states.unsqueeze(1)
        raw_surprise = ((target_states - predictions) ** 2).mean(dim=-1) 
        
        energy_weights = torch.ones(self.num_nodes, device=states.device)
        energy_weights[self.motor_idx] = 50.0  
        weighted_surprise = raw_surprise * active_weights.unsqueeze(0) * energy_weights.unsqueeze(0).unsqueeze(0)
        total_energy = weighted_surprise.sum() / (active_weights.sum() * batch_size + 1e-8)
        per_node_surprise = weighted_surprise.sum(dim=2).mean(dim=0)
        return total_energy, per_node_surprise

    @torch.no_grad()
    def partial_physics_step(self, states):
        state_corr = torch.einsum('bnd,bmd->nm', states, states) / (self.state_dim * states.shape[0])
        dist_vectors = self.coords.unsqueeze(1) - self.coords.unsqueeze(0)
        dist_matrix  = torch.cdist(self.coords, self.coords) + 1e-8
        direction    = dist_vectors / dist_matrix.unsqueeze(2)
        
        # INCREASED PULL: Hidden nodes should stretch more toward neighbors
        pull_force   = (state_corr.clamp(min=0) * self.adjacency).unsqueeze(2) * dist_vectors * 0.05
        # WEAKENED PUSH: Let nodes get a bit closer to see the structure
        push_force   = direction * (0.002 / (dist_matrix**2 + 0.5)).unsqueeze(2)
        
        # Quadratic Gravity Boundary
        center_dist = self.coords.norm(dim=1, keepdim=True)
        boundary_penalty = torch.relu(center_dist - 6.0) ** 2 * 0.2
        gravity = -self.coords * (0.005 + boundary_penalty / (center_dist + 1e-8))
        
        hidden_mask  = torch.ones(self.num_nodes, device=self.coords.device, dtype=torch.bool)
        hidden_mask[self.retina_idx] = hidden_mask[self.motor_idx] = False
        delta = (pull_force.sum(dim=1) - push_force.sum(dim=1) + gravity)
        self.coords[hidden_mask] += delta[hidden_mask]

    def relax(self, clamped_indices, clamped_values, max_steps=50, lr_state=0.5, init_states=None, capture_history=False):
        device = clamped_values.device
        batch_size = clamped_values.shape[0]
        history = []
        if init_states is not None:
            states = init_states.clone().detach()
        else:
            states = torch.rand(batch_size, self.num_nodes, self.state_dim, device=device) * 0.1
        states[:, clamped_indices, :] = clamped_values.detach()
        states.requires_grad_(True)
        for step in range(max_steps):
            if states.grad is not None: states.grad.zero_()
            energy, per_node_surprise = self.compute_masked_energy(states)
            grad = torch.autograd.grad(energy, states)[0]
            
            # 🛑 CIRCUIT BREAKER 1: The NaN Assassin
            # If the gradient exploded, instantly convert it to a safe flat number
            grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
            grad = torch.clamp(grad, min=-1.0, max=1.0)
            
            if capture_history:
                history.append({
                    'step': step,
                    'coords': self.coords.detach().cpu().numpy().copy(),
                    'surprises': per_node_surprise.detach().cpu().numpy().copy()
                })

            self.partial_physics_step(states.detach())

            with torch.no_grad():
                states.sub_(lr_state * grad)
                states.clamp_(min=-5.0, max=5.0)
                states[:, clamped_indices, :] = clamped_values.detach()
        self.current_states = states.detach()
        return self.current_states, history

    def train_step_pure_pc(self, surprise_threshold=0.5):
        if self.clamped_states is None: return 0.0, None
        self.mlp_optimizer.zero_grad()
        
        pc_loss, node_surprises = self.compute_masked_energy(self.clamped_states)
        
        # 🛑 CIRCUIT BREAKER 2: The Quarantine 
        # If the loss is poisoned, abort the training step entirely to save the weights
        if torch.isnan(pc_loss) or torch.isinf(pc_loss):
            print("⚠️ WARNING: NaN Loss Detected. Quarantining batch.")
            return 0.0, node_surprises.detach().cpu().numpy()

        if pc_loss > surprise_threshold:
            pc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.mlp_optimizer.step()
            
        return pc_loss.item(), node_surprises.detach().cpu().numpy()
    @torch.no_grad()
    def physics_step(self, lr_weight=0.01, decay_rate=0.005, min_synapses=300): 
        if self.current_states is None: return
        state_corr = torch.einsum('bnd,bmd->nm', self.current_states, self.current_states) / (self.state_dim * self.current_states.shape[0])
        self.weights[self.adjacency] *= (1 - decay_rate)
        self.weights[self.adjacency] += lr_weight * state_corr[self.adjacency].clamp(min=0.0)
        self.weights.clamp_(0.0, 1.0)
        
        center_dist = self.coords.norm(dim=1, keepdim=True)
        boundary_penalty = torch.relu(center_dist - 6.0) ** 2 * 0.2
        gravity = -self.coords * (0.005 + boundary_penalty / (center_dist + 1e-8))
        
        hidden_mask = torch.ones(self.num_nodes, device=self.coords.device, dtype=torch.bool)
        hidden_mask[self.retina_idx] = hidden_mask[self.motor_idx] = False
        self.coords[hidden_mask] += gravity[hidden_mask]

        weak = self.adjacency & (self.weights < 0.15); self.adjacency[weak] = False; self.weights[weak] = 0.0
        new_dist = torch.cdist(self.coords, self.coords); can_form = (new_dist < 3.5) & (~self.adjacency); can_form.fill_diagonal_(False)
        corr_mask = can_form & (state_corr > 0.1); self.adjacency[corr_mask] = True; self.weights[corr_mask] = 0.1
        
        curr_syn = self.adjacency.sum().item()
        if curr_syn < min_synapses:
            flat = (~self.adjacency).float(); flat.fill_diagonal_(0)
            n_needed = int(min_synapses - curr_syn)
            if flat.sum() > n_needed:
                chosen = torch.multinomial(flat.view(-1), n_needed, replacement=False)
                self.adjacency[chosen // self.num_nodes, chosen % self.num_nodes] = True
                self.weights[chosen // self.num_nodes, chosen % self.num_nodes] = 0.1