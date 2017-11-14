import numpy as np
import tensorflow as tf

from app.ds.graph.np_graph import Graph
from app.utils.constant import TRAIN, LABELS, FEATURES, SUPPORTS, MASK, VALIDATION, TEST


class DataPipeline():
    '''Class for managing the data pipeline'''

    def __init__(self, model_params, data_dir, dataset_name):
        self.graph = None
        self.populate_graph(model_params, data_dir, dataset_name)

        self.train_feed_dict = {}
        self.validation_feed_dict = {}
        self.test_feed_dict = {}
        self.populate_feed_dicts()

    def populate_graph(self, model_params, data_dir, dataset_name):
        self.graph = Graph(model_name=model_params.model_name)
        self.graph.read_data(data_dir=data_dir, datatset_name=dataset_name)
        self.graph.prepare_data(model_params=model_params)

    def populate_feed_dicts(self, dataset_splits=[0.6, 0.2, 0.2]):
        '''Method to populate the feed dicts'''
        dataset_splits_sum = sum(dataset_splits)
        dataset_splits = list(map(lambda x: x / dataset_splits_sum, dataset_splits))

        features = self.graph.features
        labels = self.graph.labels
        data_size = self.features.shape[0]
        shuffle = np.arange(data_size)
        np.random.shuffle(shuffle)
        features = features[shuffle]
        labels = labels[shuffle]

        current_index = 0
        train_index = np.arange(current_index, int(data_size * dataset_splits[0]))
        current_index = int(data_size * dataset_splits[0])
        val_index = np.arange(current_index, int(data_size * dataset_splits[1]))
        current_index = int(data_size * dataset_splits[1])
        test_index = np.arange(current_index, int(data_size * dataset_splits[2]))

        supports = self.g.compue_supports
        placeholder_dict = self.placeholder_inputs()

        feed_dict = {
            placeholder_dict[LABELS]: labels,
            placeholder_dict[FEATURES]: features,
            placeholder_dict[SUPPORTS]: supports
        }

        feed_dict[placeholder_dict[MASK]] = map_indices_to_mask(train_index, mask_size=data_size)
        self.train_feed_dict = feed_dict

        feed_dict[placeholder_dict[MASK]] = map_indices_to_mask(val_index, mask_size=data_size)
        self.validation_feed_dict = feed_dict

        feed_dict[placeholder_dict[MASK]] = map_indices_to_mask(test_index, mask_size=data_size)
        self.test_feed_dict = feed_dict


    def placeholder_inputs(self, feature_size, label_size, support_size):
        '''
        Logic borrowed from
        https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/fully_connected_feed.py'''
        labels_placeholder = tf.placeholder(tf.int32, shape=(None, label_size))
        features_placeholder = tf.placeholder(tf.float32, shape=(None, feature_size))
        support_placeholder = [tf.sparse_placeholder(tf.float32)] * support_size
        mask_placeholder = tf.placeholder(tf.bool, shape=(None, 1))
        return {
            FEATURES: features_placeholder,
            LABELS: labels_placeholder,
            SUPPORTS: support_placeholder,
            MASK: mask_placeholder
        }

    def get_feed_dict(self, mode=TRAIN):
        if mode == TRAIN:
            return self.train_feed_dict
        elif mode == VALIDATION:
            return self.validation_feed_dict
        elif mode == TEST:
            return self.test_feed_dict
        else:
            return None

def map_indices_to_mask(indices, mask_size):
    '''Method to map the indices to a mask'''
    mask = np.zeros(mask_size, dtype=np.bool)
    mask[indices] = True
    return mask