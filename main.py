import numpy as np
import seaborn as sns
import tensorflow as tf

from app.app import train_classifier
from app.app import train_encoder
from app.model.params import ModelParams
from app.utils.constant import *

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

sns.set(color_codes=True)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(DATASET_NAME, CITESEER, "Name of the dataset. Supported values are cora, pubmed, citeseer")
flags.DEFINE_string(MODEL_NAME, GCN, "Name of the model. Supported values are ff, gcn, gcn_poly, gcn_ae, gcn_vae")
flags.DEFINE_float(LEARNING_RATE, 0.01, "Initial learning rate")
flags.DEFINE_integer(EPOCHS, 200, "Number of epochs to train for")
flags.DEFINE_integer(HIDDEN_LAYER1_SIZE, 32, "Number of nodes in the first hidden layer")
flags.DEFINE_integer(HIDDEN_LAYER2_SIZE, 16, "Number of nodes in the second hidden layer. This setting is only used "
                                             "for auto encoder models.")
flags.DEFINE_float(DROPOUT, 0.5, "Dropout rate")
flags.DEFINE_float(L2_WEIGHT, 5e-4, "Weight for L2 regularization")
flags.DEFINE_integer(EARLY_STOPPING, 20, "Number of epochs for early stopping")
flags.DEFINE_string(DATA_DIR, "/Users/shagun/projects/pregel/data", "Base directory for reading the datasets")
flags.DEFINE_bool(SPARSE_FEATURES, True, "Boolean variable to indicate if the features are sparse or not")
flags.DEFINE_bool(POLY_DEGREE, 1,
                  "Degree of the Chebyshev Polynomial. This value is used only if gcn_poly model is used.")
flags.DEFINE_string(TENSORBOARD_LOGS_DIR, "", "Directory for saving tensorboard logs")
flags.DEFINE_integer(NUM_EXP, 10, "Number of times the experiment should be run before reporting the average performance")


model_params = ModelParams(FLAGS)
data_dir = FLAGS.data_dir
dataset_name = FLAGS.dataset_name

if __name__ == "__main__":
    if(model_params.model_name in [FF, GCN, GCN_POLY]):
        train_classifier.run(model_params=model_params,
                             data_dir=data_dir,
                             dataset_name=dataset_name)
    else:
        train_encoder.run(model_params=model_params,
                          data_dir=data_dir,
                          dataset_name=dataset_name)
