'''
Collection of functions to test the stats functions
'''

from app.stats import ptest
from numpy.random import normal, uniform
import numpy as np

np.random.seed(42)

def _test_normality():
    '''
    Method to test the _check_normality function
    '''

    size = 100
    dist = normal(loc = 0, scale = 1, size = size)

    assert ptest._check_normality(dist) == True, "The input distribution is gaussian."

    dist = uniform(low = -1, high=1, size=size)

    assert ptest._check_normality(dist) == False, "The input distribution is not gaussian."

def _test_guassian_comparison():
    '''
    Method to test the _comparE_gaussian function
    '''

    size = 100
    dist1 = normal(loc=0, scale=1, size=size)

    dist2 = normal(loc=0.1, scale=0.9, size=size)
    assert ptest._compare_gaussians(dist1, dist2) == True, "The input distributions are similar."

    dist2 = normal(loc=5, scale=1, size=size)
    assert ptest._compare_gaussians(dist1, dist2) == False, "The input distributions are not similar."

    dist2 = normal(loc=5, scale=5, size=size)
    assert ptest._compare_gaussians(dist1, dist2) == False, "The input distributions are not similar."


def _test_distribution_comparison():
    '''
    Method to test the _comparE_gaussian function
    '''

    size = 100
    dist1 = normal(loc=0, scale=1, size=size)

    dist2 = normal(loc=0.1, scale=0.9, size=size)
    assert ptest._compare_distributions(dist1, dist2) == True, "The input distributions are similar."

    dist2 = uniform(low=-1, high=-1, size=size)
    assert ptest._compare_gaussians(dist1, dist2) == False, "The input distributions are not similar."


if __name__ == "__main__":
    _test_normality()
    _test_guassian_comparison()
    _test_distribution_comparison()