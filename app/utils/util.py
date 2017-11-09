import numpy as np


def invert_dict(_dict):
    '''Method to invert a dict'''
    return dict([[v, k] for k, v in _dict.items()])


def map_set_to_khot_vector(index_set, num_classes):
    '''Method to map a set to a k-hot vector of `num_classes`'''
    khot_vector = np.zeros(num_classes)
    for index in index_set:
        khot_vector[index] = 1
    return khot_vector


def map_list_to_floats(item_list):
    '''Method to map a list to list of floats'''
    return np.asarray(list(map(lambda x: float(x), item_list)))
