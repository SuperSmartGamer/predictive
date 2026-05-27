import torch
from torch import nn
import torch.nn.functional as F
import uuid

newId = lambda: str(uuid.uuid4())

GLOBAL_NODE_REGISTRY = {}


class State:
    def __init__(self, node, state_dim=16):
        self.node = node
        self.value = torch.randn(state_dim)

    def decay(self):
        self.value = (1 - self.node.gamma) * self.value


class Connection(nn.Module):
    def __init__(self, from_node, to_node, direction, embedding_dim=16):
        super().__init__()
        self.from_node = from_node
        self.to_node = to_node
        self.direction = direction

        self.forward_projection = nn.Linear(embedding_dim, embedding_dim)
        self.error_projection = nn.Linear(embedding_dim, embedding_dim)

        self.trust = torch.tensor(1.0)


class Node(nn.Module):
    def __init__(self, state_dim=16):
        super().__init__()

        class SwiGLU(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = nn.Linear(in_features, out_features * 2)

            def forward(self, x):
                x, gate = self.linear(x).chunk(2, dim=-1)
                return x * F.silu(gate)

        self.id = newId()
        GLOBAL_NODE_REGISTRY[self.id] = self

        self.state_dim = state_dim

        self.network = nn.Sequential(
            SwiGLU(state_dim, 30),
            nn.Linear(30, state_dim),
            nn.LayerNorm(state_dim),
        )

        self.gamma_logit = nn.Parameter(torch.zeros(()))
        self.alpha_logit = nn.Parameter(torch.zeros(()))
        self.beta_logit = nn.Parameter(torch.zeros(()))
        self.lam_logit = nn.Parameter(torch.zeros(()))
        self.w_self_logit = nn.Parameter(torch.zeros(()))

        self.connections = nn.ModuleList()
        self.connection_map = {}
        self.incoming_grads = []
        self.state = State(self, state_dim=state_dim)
        self.surprise_threshold = 0.5

        self.Adam = torch.optim.Adam(self.parameters(), lr=0.001)

    @property
    def gamma(self):
        return torch.sigmoid(self.gamma_logit) * 0.5

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    @property
    def beta(self):
        return F.softplus(self.beta_logit) * 0.1

    @property
    def lam(self):
        return F.softplus(self.lam_logit)

    @property
    def w_self(self):
        return F.softplus(self.w_self_logit)

    def forward_network(self, x):
        return self.network(x)

    def fetch_node_from_id(self, node_id):
        return GLOBAL_NODE_REGISTRY[node_id]

    def add_connection(self, to_node_id, direction="out"):
        conn = Connection(
            from_node=self.id,
            to_node=to_node_id,
            direction=direction,
            embedding_dim=self.state_dim,
        )
        self.connections.append(conn)
        self.connection_map[to_node_id] = conn
        return conn

    def micro_step(self, lr=0.001):
        self.state.decay()

        for pg in self.Adam.param_groups:
            pg["lr"] = lr

        self.Adam.zero_grad()

        if len(self.incoming_grads) == 0:
            return

        consensus = None

        for g in self.incoming_grads:
            sender_id = g["sender"]
            if sender_id not in self.connection_map:
                continue

            conn = self.connection_map[sender_id]
            weighted = g["grads"] * conn.trust

            consensus = weighted if consensus is None else consensus + weighted

        if consensus is None:
            self.incoming_grads.clear()
            return

        consensus = consensus / (len(self.incoming_grads) + 1e-8)

        delta = self.forward_network(self.state.value)
        consensus_loss = (delta - consensus).pow(2).mean()

        l1_penalty = self.lam * sum(p.abs().sum() for p in self.network.parameters())
        loss = consensus_loss + l1_penalty

        loss.backward()
        self.Adam.step()

        self.state.value = self.state.value - delta.detach()
        self.incoming_grads.clear()

    def meso_step(self, lr=0.001):
        self.state.decay()

        for pg in self.Adam.param_groups:
            pg["lr"] = lr

        self.Adam.zero_grad()

        total_loss = torch.zeros(
            (),
            device=self.state.value.device,
            dtype=self.state.value.dtype,
        )

        num_connections = max(len(self.connections), 1)
        to_prune = []

        for connection in self.connections:
            target_node = self.fetch_node_from_id(connection.to_node)

            target_state = target_node.state.value
            predicted_state = connection.forward_projection(self.state.value)

            raw_error = target_state - predicted_state

            recon_loss = (
                1 - F.cosine_similarity(target_state, predicted_state, dim=0)
            ) + 0.1 * raw_error.pow(2).mean()

            correction_msg = connection.error_projection(raw_error)
            stabilization_loss = 0.05 * F.mse_loss(
                correction_msg,
                raw_error.detach()
            )

            total_loss += recon_loss + stabilization_loss

            trust_target = torch.exp(-recon_loss.detach())
            connection.trust = (1 - self.alpha) * connection.trust + self.alpha * trust_target

            if connection.trust.item() < self.beta.item():
                to_prune.append(connection)

            if recon_loss.detach().item() > self.surprise_threshold:
                self.send_gradients(connection.to_node, correction_msg)

        for connection in to_prune:
            self.connections.remove(connection)
            self.connection_map.pop(connection.to_node, None)

        total_loss = total_loss / (num_connections + 1e-8)
        total_loss.backward()
        self.Adam.step()

    def send_gradients(self, to_node_id, gradients):
        node = self.fetch_node_from_id(to_node_id)
        node.incoming_grads.append({
            "sender": self.id,
            "grads": gradients
        })