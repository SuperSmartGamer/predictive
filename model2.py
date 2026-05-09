import torch
import torch.nn as nn
import math

class SpatialDynamicGraph(nn.Module):
    def __init__(self, num_nodes=150, state_dim=16, space_dim=3):
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

        self.register_buffer('coords', torch.randn(num_nodes, space_dim) * 1.5)
        self.register_buffer('weights', torch.zeros(num_nodes, num_nodes))
        self.register_buffer('freqs', torch.zeros(num_nodes, num_nodes))
        self.register_buffer('adjacency', torch.zeros(num_nodes, num_nodes, dtype=torch.bool))

        self.retina_idx = list(range(0, 10))
        self.motor_idx  = list(range(10, 20))
        self.hidden_idx = list(range(20, num_nodes))

        retina = [[3.0 * math.cos(2*math.pi*i/10), 3.0 * math.sin(2*math.pi*i/10), -4.0] for i in range(10)]
        motor  = [[3.0 * math.cos(2*math.pi*i/10), 3.0 * math.sin(2*math.pi*i/10),  4.0] for i in range(10)]
        self.register_buffer('retina_anchors', torch.tensor(retina, dtype=torch.float32))
        self.register_buffer('motor_anchors',  torch.tensor(motor,  dtype=torch.float32))

        with torch.no_grad():
            self.coords[self.retina_idx] = self.retina_anchors
            self.coords[self.motor_idx]  = self.motor_anchors

        self._init_random_topology(density=0.20)
        self.mlp_optimizer = torch.optim.Adam(self.parameters(), lr=0.002, weight_decay=1e-5)
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
        h1 = torch.nn.functional.layer_norm(h1, [h1.size(-1)])
        h1 = torch.nn.functional.leaky_relu(h1)
        h2 = torch.einsum('bnmh,nhd->bnmd', h1, self.W2) + self.b2.unsqueeze(1)
        return torch.tanh(h2)

    def compute_masked_energy(self, states):
        batch_size = states.shape[0]
        active_weights = self.weights * self.adjacency.float()

        total_routed   = torch.einsum('nm,bmd->bnd', active_weights, states)
        direct_messages = active_weights.unsqueeze(0).unsqueeze(3) * states.unsqueeze(1)
        masked_context  = total_routed.unsqueeze(2) - direct_messages

        rel_pos       = (self.coords.unsqueeze(0) - self.coords.unsqueeze(1)).detach()
        rel_pos_batch = rel_pos.unsqueeze(0).expand(batch_size, -1, -1, -1)

        mlp_input  = torch.cat([masked_context, rel_pos_batch], dim=-1)
        predictions = self.mlp_forward(mlp_input)
        target_states = states.unsqueeze(1)

        raw_surprise = ((target_states - predictions) ** 2).mean(dim=-1)

        energy_weights = torch.ones(self.num_nodes, device=states.device)
        energy_weights[self.motor_idx] = 10.0

        weighted_surprise = raw_surprise * active_weights.unsqueeze(0) * energy_weights.unsqueeze(0).unsqueeze(0)
        total_energy = weighted_surprise.sum() / (active_weights.sum() * batch_size + 1e-8)

        degree = active_weights.sum(dim=1).clamp(min=1.0)
        per_node_surprise = (weighted_surprise.sum(dim=2).mean(dim=0) / degree)
        raw_per_node = raw_surprise.mean(dim=0).mean(dim=1)

        return total_energy, per_node_surprise, raw_per_node

    @torch.no_grad()
    def partial_physics_step(self, states):
        state_corr = torch.einsum('bnd,bmd->nm', states, states) / (self.state_dim * states.shape[0])
        
        # THE SYMMETRY BREAKER: Injecting 1e-4 noise to prevent point collapse (0 distance)
        dist_vectors = self.coords.unsqueeze(1) - self.coords.unsqueeze(0)
        dist_vectors += torch.randn_like(dist_vectors) * 1e-4 
        
        dist_matrix = torch.norm(dist_vectors, dim=2) + 1e-8
        direction   = dist_vectors / dist_matrix.unsqueeze(2)

        degree     = self.adjacency.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        pull_force = (state_corr.clamp(min=0) * self.adjacency).unsqueeze(2) * dist_vectors * (0.05 / degree.unsqueeze(2))
        push_force = direction * (0.02 / (dist_matrix**2 + 1e-3)).unsqueeze(2)

        center_dist      = self.coords.norm(dim=1, keepdim=True)
        boundary_penalty = torch.relu(center_dist - 6.0) ** 2 * 0.2
        gravity          = -self.coords * (0.005 + boundary_penalty / (center_dist + 1e-8))
        thermal_noise    = torch.randn_like(self.coords) * 0.002

        hidden_mask = torch.ones(self.num_nodes, device=self.coords.device, dtype=torch.bool)
        hidden_mask[self.retina_idx] = False
        hidden_mask[self.motor_idx]  = False

        delta = pull_force.sum(dim=1) - push_force.sum(dim=1) + gravity + thermal_noise
        delta = torch.nan_to_num(delta, nan=0.0)
        delta = torch.clamp(delta, min=-0.02, max=0.02)
        self.coords[hidden_mask] += delta[hidden_mask]

        # Symmetry breaker applied to Pauli Exclusion as well
        for _ in range(3):
            d_vecs = self.coords.unsqueeze(1) - self.coords.unsqueeze(0)
            d_vecs += torch.randn_like(d_vecs) * 1e-4
            d_mat = torch.norm(d_vecs, dim=2) + 1e-8
            dir_mat = d_vecs / d_mat.unsqueeze(2)
            
            overlap = torch.relu(0.5 - d_mat)
            overlap.fill_diagonal_(0)
            correction = (dir_mat * overlap.unsqueeze(2)).sum(dim=1) * 0.5
            self.coords[hidden_mask] += correction[hidden_mask]

    def continuous_step(self, clamped_indices, clamped_values, lr_state=0.5):
        if self.current_states is None:
            batch_size = clamped_values.shape[0]
            self.current_states = torch.rand(batch_size, self.num_nodes, self.state_dim, device=clamped_values.device) * 0.1

        states = self.current_states.detach().requires_grad_(True)

        with torch.no_grad():
            states[:, clamped_indices, :] = clamped_values

        energy, per_node_surprise, raw_per_node = self.compute_masked_energy(states)
        grad = torch.autograd.grad(energy, states)[0]

        grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
        grad = torch.clamp(grad, min=-1.0, max=1.0)

        with torch.no_grad():
            states.sub_(lr_state * grad)
            states.clamp_(min=-5.0, max=5.0)
            states[:, clamped_indices, :] = clamped_values

            hidden_states = states[:, self.hidden_idx, :]
            var = hidden_states.var()
            if var < 0.01:
                states[:, self.hidden_idx, :] += torch.randn_like(hidden_states) * 0.1

        self.partial_physics_step(states.detach())
        self.current_states = states
        return energy.item(), per_node_surprise.detach().cpu(), raw_per_node.detach().cpu()

    def inference_step(self, clamped_indices, clamped_values, states, lr_state=0.5):
        states = states.detach().requires_grad_(True)
        with torch.no_grad():
            states[:, clamped_indices, :] = clamped_values

        energy, _, _ = self.compute_masked_energy(states)
        grad = torch.autograd.grad(energy, states)[0]

        grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
        grad = torch.clamp(grad, min=-1.0, max=1.0)

        with torch.no_grad():
            states.sub_(lr_state * grad)
            states.clamp_(min=-5.0, max=5.0)
            states[:, clamped_indices, :] = clamped_values

        return states

    def train_step_pure_pc(self, surprise_threshold=0.1):
        if self.current_states is None:
            return 0.0, None, None
        self.mlp_optimizer.zero_grad()
        pc_loss, node_surprises, raw_surprises = self.compute_masked_energy(self.current_states)

        if torch.isnan(pc_loss) or torch.isinf(pc_loss):
            return 0.0, node_surprises.detach().cpu(), raw_surprises.detach().cpu()

        if pc_loss > surprise_threshold:
            pc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.mlp_optimizer.step()

        return pc_loss.item(), node_surprises.detach().cpu(), raw_surprises.detach().cpu()

    @torch.no_grad()
    def physics_step(self, lr_weight=0.01, decay_rate=0.005, min_synapses=300):
        if self.current_states is None: return
        state_corr = torch.einsum('bnd,bmd->nm', self.current_states, self.current_states) / (self.state_dim * self.current_states.shape[0])
        
        self.weights[self.adjacency] *= (1 - decay_rate)
        self.weights[self.adjacency] += lr_weight * state_corr[self.adjacency].clamp(min=0.0)
        self.weights.clamp_(0.0, 1.0)

        weak = self.adjacency & (self.weights < 0.15)
        self.adjacency[weak] = False
        self.weights[weak]   = 0.0

        new_dist = torch.cdist(self.coords, self.coords)
        can_form = (new_dist < 3.0) & (~self.adjacency)
        can_form.fill_diagonal_(False)
        corr_mask = can_form & (state_corr > 0.1)
        self.adjacency[corr_mask] = True
        self.weights[corr_mask]   = 0.1

        curr_syn = self.adjacency.sum().item()
        if curr_syn < min_synapses:
            flat = (~self.adjacency).float()
            flat.fill_diagonal_(0)
            n_needed = int(min_synapses - curr_syn)
            if flat.sum() > n_needed:
                chosen = torch.multinomial(flat.view(-1), n_needed, replacement=False)
                self.adjacency[chosen // self.num_nodes, chosen % self.num_nodes] = True
                self.weights[chosen // self.num_nodes, chosen % self.num_nodes]   = 0.1