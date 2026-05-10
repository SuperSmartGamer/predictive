from model import *
import random
from pprint import pprint
from gui_helper import *
import numpy as np
graph=TiledGraph()

for _ in range(100):
    graph.create_node(Pos(random.randint(0, 100), random.randint(0, 100)))
graph.bucket_size=10
graph.bucketize()

c=graph.nodes.values()

for x in c:
    print(x)
    break