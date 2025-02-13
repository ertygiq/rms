"""
================================================================================
Custom Clustering Score My Cool Metric: Redundancy & Missing (for picking one item per cluster)
================================================================================

[GPT-generated]

Overview
--------
This module defines a custom metric to evaluate clustering results under the
specific scenario where we ultimately pick exactly one representative item from
each predicted cluster.

Key Ideas
---------
1) We want each true cluster to appear exactly once in the final selection.
2) Two main penalties arise:
   - Missing: A true cluster does not appear in any predicted cluster.
   - Redundancy: A true cluster is covered by more than one predicted cluster,
     causing a potential for multiple picks from the same true cluster.

Calculation Steps
-----------------
1) Probability Computation:
   - For each true cluster T, identify each predicted cluster pr_cl that
     intersects T.
   - Let 'theta' be the fraction of pr_cl items that belong to T:
       theta = (number of items in (T, pr_cl)) / (total number of items in pr_cl)

2) Missing Probability (for a single true cluster T):
   - We calculate the probability that T is "missed" by all predicted clusters
     that intersect T:
       prob_missed_T = product of (1 - theta_i) over all relevant predicted clusters
     This is the chance that, if you pick one item at random from each predicted
     cluster, you never pick anything from T.

3) Redundancy (for a single true cluster T):
   - We calculate how "overlapped" T is by predicted clusters:
       redundancy_T = (sum of theta_i) - 1 + prob_missed_T
     To interpret the formula, note that each predicted cluster’s contribution to picking an item from true cluster T
     follows a Bernoulli trial with parameter “theta,” yielding a Poisson-Binomial distribution when combined.
     The expected number of items chosen from T is the sum of all theta values. However, our metric focuses on
     the scenario where “redundancy” is the expected count above one item. Hence, we take “sum of thetas - 1.”
     But in doing so, we would inadvertently push the zero-count case below zero, so we add back the probability
     of that zero-count scenario (the probability of missing T entirely) to correct for it.

4) Summation Across All True Clusters:
   - For each true cluster, we get an expected "missing" value and an expected
     "redundancy" value.
   - By the linearity of expectation, we simply sum these values across all
     true clusters to get:
       total_missing
       total_redundancy

5) Final Score:
   - We combine the total redundancy and total missing with a weight
     (losing_weight) applied to missing:
       score = total_redundancy + losing_weight * total_missing
   - The idea is that missing an entire cluster should be penalized more
     severely (hence the multiplier).

Why This Metric?
----------------
Unlike standard metrics (Rand Index, Purity, NMI, etc.), this custom metric is
directly tailored to the case where you only pick one item from each predicted
cluster. It penalizes both missing clusters and overlapping clusters that lead
to redundant picks.

Below is the actual code implementing the logic described.
"""

import math
import numpy as np
import pandas as pd


# Calculates number of pr_cl labels in the t_cl. To make it quicly
# the sorted zipped labels are provided as well as index i from which to start counting.
def num_of_pr_cl_in_curr_t_cl(i, t_cl, pr_cl, zipped):
    num = 0
    total_len = len(zipped)
    while True:
        curr_t_lb, curr_pr_lb = zipped[i]

        if curr_t_lb != t_cl or curr_pr_lb != pr_cl:
            # so if new true cluster starts, we finish
            # or if still same true cluster but new pred cluster, stop
            break

        if curr_t_lb == t_cl and curr_pr_lb == pr_cl:
            num += 1
        i += 1
        if i >= total_len:
            break
    return num


# Get the total number of elements in pred cluster pr_cl
def get_pr_total_num(sorted_pred_lbs, pr_cl):
    return sorted_pred_lbs.count(pr_cl)


# possibilities is the list of probabilities
# Imagine for true cluster 1 there are 2 pred clusters (A and B) for this items
# So this list would have to elements. First - probability of element of A pred cluster to be in this true cluster
# And the second - probability of element of B pred cluster to be in this true cluster.
# And this not just for 2, for any number.
#
# The return value is a tuple: expected value of missing this true cluster, and expected value of number of
# redundant elements from this cluster
def get_expectations(possibilities):
    # probability that this cluster would be missed
    prob_all_zeros = math.prod(1 - theta for theta in possibilities)
    # So here is idea: I treat the list of possibilities as theta params of Bernoulli random vars.
    # of if X var is total number of ones, then E[X - 1] + prob_all_zeros is what I'm looking for.
    redundant_exp = sum(possibilities) - 1 + prob_all_zeros
    return redundant_exp, prob_all_zeros


def prepare_lbs(lbs):
    # lbs can be Seried, np array or list
    res = pd.Series(lbs).fillna(-1).astype(int).values
    if -1 in res:
        res = unclust_to_seq_clusters(res)
    return np.array(res)

# 0 - is ideal
# The bigger the number - the worse
def my_cool_metric(true_lbs, pred_lbs, losing_weight=3):
    if len(true_lbs) != len(pred_lbs):
        raise Exception('Lists must be of the same size')
    true_lbs = prepare_lbs(true_lbs)
    pred_lbs = prepare_lbs(pred_lbs)
    # Todo replace -1 labels
    zipped = list(zip(true_lbs, pred_lbs))

    # NEVER EVER do this. This will break zipped object
    # list(zipped)

    sorted_pred_lbs = sorted(pred_lbs)  # for get_pr_total_num
    # sorting first on a true labels columns and then by pred label column.
    # The algo below relies on this logic
    zipped = sorted(zipped, key=lambda x: (x[0], x[1]))
    # the key is a cluster index in preds. The value is the size of this cluster
    # Used for caching
    total_num_of_pred_cls = {}

    expectations_per_cluster = []

    # pointing to the cluster currently being processed
    curr_t_cl = None
    processed_pr_cls = []
    curr_cl_possibilities = []
    for i, (t_cl, pr_cl) in enumerate(zipped):
        # initial value
        if curr_t_cl is None:
            curr_t_cl = t_cl

        if t_cl != curr_t_cl:
            # new true cluster started
            expectations_per_cluster.append(get_expectations(curr_cl_possibilities))
            processed_pr_cls = []
            curr_cl_possibilities = []
            curr_t_cl = t_cl

        # skipping pred cluster if already processed it for current true cluster
        if pr_cl in processed_pr_cls:
            continue
        num_here = num_of_pr_cl_in_curr_t_cl(i, t_cl, pr_cl, zipped)
        if pr_cl not in total_num_of_pred_cls:
            total_num_of_pred_cls[pr_cl] = get_pr_total_num(sorted_pred_lbs, pr_cl)
        num_total = total_num_of_pred_cls[pr_cl]
        # so storing ratio of pred cluster items in this cluster to the total number of items in this pred cluster
        # This is a probability that the picked item from that pred cluster would be picked in this cluster
        curr_cl_possibilities.append(num_here / num_total)
        processed_pr_cls.append(pr_cl)

    # process last cluster
    expectations_per_cluster.append(get_expectations(curr_cl_possibilities))
    expectations_per_cluster = np.array(expectations_per_cluster)
    redund_exp, missing_exp = expectations_per_cluster.sum(axis=0)
    score = redund_exp + losing_weight * missing_exp
    # print((redund_exp, missing_exp, score)) # for debug
    return score


# Conversts unclustered labels (-1) to sequential clusters labels: max, max+1, ...
def unclust_to_seq_clusters(labels):
    arr = np.array(labels.copy())
    max_val = arr.max()
    # Find all indices where the value is -1
    neg_ones = np.where(arr == -1)[0]
    # Replace each -1 with sequential values starting from max_val + 1
    arr[neg_ones] = np.arange(max_val + 1, max_val + 1 + len(neg_ones))
    return arr


# Some example
tl = [1, 1, 2, 2, 3, 3]
pl = [2, 1, 3, 2, 3, 4]
res = my_cool_metric(tl, pl)
print(res)

# Correct is (np.float64(0.3200000000000001), np.float64(1.32), np.float64(4.28))
tl = [1,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4]
pl = [1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3]
res = my_cool_metric(tl, pl)
print(res)

# same as above. Shows it's indifferent to label names
tl = [10,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4]
pl = [1,1,2,2,2,2,2,2,2,2,2,2,5,5,5,5,5]
res = my_cool_metric(tl, pl)
print(res)

# swapped the places. Moral here is that score is differnt depending on who is ground truth labels
# This is correct: (np.float64(1.0285714285714285), np.float64(0.028571428571428574), np.float64(1.114285714285714))
tl = [1,1,2,2,2,2,2,2,2,2,2,2,5,5,5,5,5]
pl = [10,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4]
res = my_cool_metric(tl, pl)
print(res)

# Exact match
tl = [1,2,3,4,5, None]
pl = [7,8,9,10, 14, -1]
res = my_cool_metric(tl, pl)
print(res)
