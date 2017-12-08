from scipy.stats import mannwhitneyu
from scipy.stats import normaltest
from scipy.stats import ttest_ind


def _check_normality(dist):
    '''
    Method to check if the samples in dist differs from a normal distribution.
    Return true if the dist is likely to be gaussian.
    '''
    _, pvalue = normaltest(dist)
    if (pvalue > 0.05):
        # The samples in dist came from a normal distribution with 95% confidence
        return True
    else:
        return False


def _compare_gaussians(dist1, dist2):
    '''Method to compare samples from two gaussians distributions to determine if they are likely to be drawn from the
     same distribution.
    Here we assume that we already know that the 2 dist are indeed gaussians.
    Return true if the two list of samples are likely to be drawn from the same gaussian'''

    _, pvalue = ttest_ind(dist1, dist2, equal_var=False)

    if (pvalue > 0.05):
        # Likely to be drawn from same gaussian
        return True
    else:
        return False


def _compare_distributions(dist1, dist2):
    '''Method to compare samples from two general distributions to determine if they are likely to be drawn from the
     same distribution.
    Return true if the two list of samples are likely to be drawn from the same distribution'''

    _, pvalue = mannwhitneyu(dist1, dist2)

    if (pvalue > 0.05):
        # Likely to be drawn from same distribution
        return True

    else:
        return False


def is_significant(dist1, dist2):
    '''Method to check if the difference between the values from two distributions is significant.
    Return True if the difference is signficant.
    '''
    if (_check_normality(dist1) and _check_normality(dist2)):
        return not _compare_gaussians(dist1, dist2)
    else:
        return not _compare_distributions(dist1, dist2)
