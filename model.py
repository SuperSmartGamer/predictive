import torch
from torch import nn
import uuid


newId=lambda: str(uuid.uuid4())

class Pos():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def to_tuple(self):
        return (self.x, self.y)


class Connection(nn.Module):
    def __init__(self, to_node,embedding_dim):
        super(Connection, self).__init__()
        self.to_node = to_node
        self.weight = nn.Parameter(torch.randn(embedding_dim,embedding_dim))
    def forward(self, input_state):
        # Placeholder
        return input_state @ self.weight
    

class State():
    def __init__(self, node):
        self.node = node
        self.value = torch.zeros(node.state_dim)
        self.shape=[16]

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

    def forward(self, input_state):
        
        return input_state
    
    def get_bucket(self, bucket_size):
        return Pos(int(self.pos.x // bucket_size), int(self.pos.y // bucket_size))
    
    def add_connection(self, to):
        self.connections.append(Connection(to,self.state_dim))

class TiledGraph(nn.Module):
    def __init__(self):
        super(TiledGraph, self).__init__()
        self.nodes = {}
        self.buckets = {}
        self.num_nodes = 0
        self.state_dim = 16
        self.bucket_size = 5

        
    def create_node(self, pos=[0,0]):
        id=newId()
        node=Node(id=id, pos=Pos(x=pos[0], y=pos[1])) if type(pos)==list else Node(id=id, pos=pos)
        self.nodes[id] = node
        self.num_nodes += 1


    def bucketize(self):
        for v in self.nodes.values():
            c = v.get_bucket(self.bucket_size)
            key = c.to_tuple()

            if key not in self.buckets:
                self.buckets[key] = set()
            
            self.buckets[key].add(v.id)
            #self.buckets[v.id]=key
            v.bucket=key

class Model(nn.Module):
    def __init__(self, node_count:int):
        super().__init__()
        self.graph=TiledGraph()

        for x in range(node_count):
            self.graph.create_node(torch.randn(0,100))
        for key in self.graph.nodes.keys():
            for node in self.graph.nodes.values():
                node.add_connection(key)
            