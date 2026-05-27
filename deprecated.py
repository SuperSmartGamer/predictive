    






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
