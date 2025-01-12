# Redundancy & Missing Score (RMS) - A Novel Clustering Score

## Overview  
This module defines a custom metric to evaluate clustering results under the specific scenario where we ultimately pick exactly one representative item from each predicted cluster.

## Key Ideas  
1. We want each true cluster to appear exactly once in the final selection.  
2. Two main penalties arise:  
   - **Missing**: A true cluster does not appear in any predicted cluster.  
   - **Redundancy**: A true cluster is covered by more than one predicted cluster, causing a potential for multiple picks from the same true cluster.  

## Calculation Steps  

### 1. Probability Computation  
For each true cluster `T`, identify each predicted cluster `pr_cl` that intersects `T`.  
Let `θ` (theta) be the fraction of `pr_cl` items that belong to `T`:  
`θ = (number of items in (T, pr_cl)) / (total number of items in pr_cl)`

### 2. Missing Probability (for a single true cluster T)  
Calculate the probability that `T` is "missed" by all predicted clusters that intersect `T`:  
`prob_missed_T = product of (1 - θ_i) over all relevant predicted clusters`  
This is the chance that, if you pick one item at random from each predicted cluster, you never pick anything from `T`.

### 3. Redundancy (for a single true cluster T)  
Calculate how "overlapped" `T` is by predicted clusters:  
`redundancy_T = (sum of θ_i) - 1 + prob_missed_T`

#### Interpretation:  
- Each predicted cluster’s contribution to picking an item from true cluster `T` follows a Bernoulli trial with parameter `θ`, yielding a Poisson-Binomial distribution when combined.  
- The expected number of items chosen from `T` is the sum of all `θ` values.  
- The metric focuses on "redundancy," the expected count above one item, calculated as `(sum of θ - 1)`.  
- To correct for pushing the zero-count case below zero, the probability of missing `T` entirely (`prob_missed_T`) is added back.  

### 4. Summation Across All True Clusters  
For each true cluster, calculate the expected "missing" and "redundancy" values.  
By the linearity of expectation, sum these values across all true clusters to get:  
- `total_missing`  
- `total_redundancy`

### 5. Final Score  
Combine the total redundancy and total missing with a weight (`losing_weight`) applied to missing:  
`score = total_redundancy + losing_weight * total_missing`  

Missing an entire cluster is penalized more severely, hence the multiplier.

## Why This Metric?  
Unlike standard metrics (Rand Index, Purity, NMI, etc.), this custom metric is directly tailored to the case where you only pick one item from each predicted cluster. It penalizes both:  
1. Missing clusters.  
2. Overlapping clusters that lead to redundant picks.

## Examples  

### Example 1: Basic Usage
`tl = [1, 1, 2, 2, 3, 3]`  
`pl = [2, 1, 3, 2, 3, 4]`  
`res = my_cool_metric(tl, pl)`  
`print(res)`

Output:  
`2.0`
---

### Example 2: Exact Match  
`tl = [1, 2, 3, 4, 5]`  
`pl = [7, 8, 9, 10, 14]`  
`res = my_cool_metric(tl, pl)`  
`print(res)`  

Output:  
`0.0`
---

### Example 3: Larger Clusters  
`tl = [1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4]`  
`pl = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]`  
`res = my_cool_metric(tl, pl)`  
`print(res)`   

Output:  
`4.28`
---

### Example 4: Indifference to Label Names  
`tl = [10, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4]`  
`pl = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]`  
`res = my_cool_metric(tl, pl)`  
`print(res)`

Output:  
`4.28`
---

### Example 5: Effect of Swapping Ground Truth and Predicted Labels  
`pl = [10, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4]`  
`tl = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5]`  
`res = my_cool_metric(tl, pl)`  
`print(res)`  

Output:  
`1.114`
