import numpy as np
import tensorflow as tf

from app.ds.data_pipeline import DataPipeline
from app.model.gcn_model import Model as GCNModel
from app.model.params import ModelParams, SparseModelParams
from app.utils.constant import *

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

from time import time
from tensorflow.contrib.keras import backend as K

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(DATASET_NAME, CORA, "Name of the dataset. Supported values are cora")
flags.DEFINE_string(MODEL_NAME, GCN, "Name of the model. Supported values are ff, gcn")
flags.DEFINE_float(LEARNING_RATE, 0.01, "Initial learning rate")
flags.DEFINE_integer(EPOCHS, 200, "Number of epochs to train for")
flags.DEFINE_integer(HIDDEN_LAYER1_SIZE, 16, "Number of nodes in the first hidden layer")
flags.DEFINE_float(DROPOUT, 0.0, "Dropout rate")
flags.DEFINE_float(L2_WEIGHT, 1e-4, "Weight for L2 regularization")
flags.DEFINE_integer(EARLY_STOPPING, 20, "Number of epochs for early stopping")
flags.DEFINE_string(DATA_DIR, "/Users/shagun/projects/pregel/data", "Base directory for reading the datasets")
flags.DEFINE_bool(SPARSE_FEATURES, True, "Boolean variable to indicate if the features are sparse or not")
flags.DEFINE_bool(SUPPORT_SIZE, 1, "Number of supports to be used")

model_params = ModelParams(FLAGS)
data_dir = FLAGS.data_dir
dataset_name = FLAGS.dataset_name

datapipeline = DataPipeline(model_params=model_params,
                            data_dir=data_dir,
                            dataset_name=dataset_name)

placeholder_dict = datapipeline.get_placeholder_dict()

feed_dict_train = datapipeline.get_feed_dict(mode=TRAIN)

feed_dict_val = datapipeline.get_feed_dict(mode=VALIDATION)

feed_dict_test = datapipeline.get_feed_dict(mode=TEST)

sparse_model_params = SparseModelParams(
    num_elements=datapipeline.num_elements,
    feature_size=datapipeline.feature_size
)
#
# model = FFModel(model_params=model_params, sparse_model_params=sparse_model_params, placeholder_dict = placeholder_dict)

model = GCNModel(model_params=model_params, sparse_model_params=sparse_model_params, placeholder_dict=placeholder_dict)

sess = tf.Session()
K.set_session(sess)
sess.run(tf.global_variables_initializer())

for epoch in range(model_params.epochs):
    start_time = time()
    loss, accuracy, opt = sess.run([model.loss, model.accuracy, model.optimizer_op], feed_dict=feed_dict_train)
    loss_val, accuracy_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_test)
    print(accuracy_val)
