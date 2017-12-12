import numpy as np


def invert_dict(_dict):
    '''Method to invert a dict'''
    return dict([[v, k] for k, v in _dict.items()])


def map_set_to_khot_vector(index_set, num_classes):
    '''Method to map a set to a k-hot vector of `num_classes`'''
    khot_vector = np.zeros(num_classes, dtype=np.int32)
    for index in index_set:
        khot_vector[index] = 1
    return khot_vector


def map_list_to_floats(item_list):
    '''Method to map a list to list of floats'''
    return np.asarray(list(map(lambda x: float(x), item_list)))

def get_class_variables(class_variable):
    '''Method to return the class variables as a dict.
    Taken from: https://stackoverflow.com/questions/21322244/getting-a-dictionary-of-class-variables-and-values'''
    return {key:value for key, value in class_variable.__dict__.items()
            if not key.startswith('__') and not callable(key)}