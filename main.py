from app.ds.graph.np_graph import Graph
from app.utils.constant import *
from app.model.params import ModelParams

import tensorflow as tf
import numpy as np

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(DATASET, CORA, "Name of the dataset. Supported values are CORA")
flags.DEFINE_string(MODEL_NAME, GCN, "Name of the model. Supported values are GCN")
flags.DEFINE_float(LEARNING_RATE, 0.01, "Initial learning rate")
flags.DEFINE_integer(EPOCHS, 200, "Number of epochs to train for")
flags.DEFINE_integer(HIDDEN_LAYER1_SIZE, 16, "Number of nodes in the first hidden layer")
flags.DEFINE_float(DROPOUT, 0.5, "Dropout rate")
flags.DEFINE_float(L2_WEIGHT, 5e-4, "Weight for L2 regularization")
flags.DEFINE_integer(EARLY_STOPPING, 20, "Number of epochs for early stopping")

model_name= FLAGS.model_name
model_params = ModelParams(model_name=model_name)

data_dir = "/Users/shagun/projects/pregel/data"
dataset_name = "cora"

g = Graph(model_name = model_name)

g.read_data(data_dir=data_dir, datatset_name=dataset_name)

g.prepare_data(model_params=model_params)