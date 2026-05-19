from xml.parsers.expat import model

import torch
from torch import nn
import uuid
import torch.nn.functional as F
from dataclasses import dataclass

newId=lambda: str(uuid.uuid4())


class Connection(nn.Module):
    def __init__(self, to_node,embedding_dim):
        super(Connection, self).__init__()
        self.to_node = to_node
        self.weight = nn.Parameter(torch.randn(embedding_dim,embedding_dim))
    def forward(self, input_state):
        # Placeholder
        return input_state @ self.weight
    

class State():
    def __init__(self, node, gamma):
        self.node = node
        self.value = torch.randn(node.state_dim)
        self.shape=[16]
        self.gamma = gamma

        def decay(self, gamma=self.gamma):
            self.value *= gamma
"""
class Node(nn.Module):
    def __init__(self,id,pos):
        super(Node, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 30),
            nn.ReLU(),
            nn.Linear(30, 16),
            nn.LayerNorm(16)
        )
        
        self.id=id
        self.connections =[]
        self.pos=pos
        self.state_dim = 16
        self.state = State(self)
        self.stateProjection = nn.Linear(self.state.shape[0], 16)
        self.bucket=[]
        
    def calculate_Loss():
        pass

    def forwardNetwork(self, input_state):
        out=self.stateProjection(input_state)
        return self.network(out+input_state)
    
    def updateState(self, input_state):
        self.state.value = self.forwardNetwork(input_state)

"""


class ModelSpace(max_Nodes=10,connections_per_node=4, embedding_dim=16):
    def __init__(self):
        self.node_indices = []
        self.weights=[]
@dataclass        
class Learnables(slots=True):
    alpha: 0.0
    beta: 0.0
    gamma: 0.0 # decay rate for state
    lamda_penalty: 0.0 
    w_self: 0.0 #weight for self when calculating gradients
        
class Node(nn.Module):
    def __init__(self):
        super(Node, self).__init__()
        class SwiGLU(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = nn.Linear(in_features, out_features * 2)

            def forward(self, x):
                x, gate = self.linear(x).chunk(2, dim=-1)
                return x * F.silu(gate)
            
        self.network = nn.Sequential(
            SwiGLU(16, 30),
            nn.Linear(30, 16),
            nn.LayerNorm(16)
            )
        self.Adam = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.incoming_grads = [
            torch.zeros_like(p)
            for p in self.network.parameters()]
        
        self.learnables = Learnables()
        self.state = State(self, self.learnables.gamma)
        self.neighbors = []
        self.weights = {}
        self.projections = []

    def fetch_node_from_id(self, node_id):
        # Placeholder for fetching a node by its ID
        return None
    
    #returns a dict of neighbor_id:neighbor_state
    def fetch_neighbor_states(self):
        states={}
        for neighbor in self.neighbors:
            node = self.fetch_node_from_id(neighbor)
            states[neighbor] = node.state.value

        return states



    def micro_step(self, lr=0.001):
        self.state.decay()       
        self.Adam.zero_grad()
        self.network.zero_grad()
        consensus=torch.stack(list(self.fetch_neighbor_states().values())).mean(dim=0)
        self.state.value-=self.forwardNetwork(self.state.value)
        loss=0
        for neighbor,projection in zip(self.neighbors,self.projections):
            n=self.fetch_node_from_id(neighbor).state.value
            c=projection(self.state.value)
            a=1 - F.cosine_similarity(n,c, dim=1)
            a=a.mean()
            loss+=a
        loss.backward()


    #idek bro
    def internal_step(self, lr=0.001):
        self.state.decay()       
        self.Adam.zero_grad()
        self.network.zero_grad()

        self.state.value-=self.network(self.state.value)
        loss=0
        for neighbor,projection in zip(self.neighbors,self.projections):
            n=self.fetch_node_from_id(neighbor).state.value
            c=projection(self.state.value)
            a=1 - F.cosine_similarity(n,c, dim=1)
            a=a.mean()
            loss+=a
        loss.backward()

       
        with torch.no_grad():
            for p, g in zip(self.network.parameters(), self.incoming_grads):
                if p.grad is None:
                    p.grad = g["grads"] * self.weights[g["sender"]]
                else:
                    p.grad += g["grads"] * self.weights[g["sender"]]
        

        self.incoming_grads=[torch.zeros_like(p) for p in self.network.parameters()]
        self.Adam.step(lr=lr)

    def send_gradients(self, target_node_id, projection, grad):
        
        for 
        
        
        pass


class Model(nn.Module):
    def __init__(self, node_count:int):
        super().__init__()
        pass

            