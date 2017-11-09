from app.ds.graph.np_graph import Graph
from app.utils.constant import GCN

# This section would be eventually replaced by argparams and configs
model_name= GCN
data_dir = "/Users/shagun/projects/pregel/data"
dataset_name = "cora"

g = Graph(model_name = model_name)

g.read_data(data_dir=data_dir, datatset_name=dataset_name)
