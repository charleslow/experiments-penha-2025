# Table 2: R@30 of the joint generative model using different Semantic ID construction methods

R@30 of the joint generative model using different Semantic ID construction methods that consider both objectives (search and recommendation). **Head** indicates the effectiveness for the top 1% most popular items in the train set, where **Torso** is the remaining set of items. Search data does not have a popularity bias, i.e. all items have the same number of queries. **Bold** indicates highest scores, while underline indicates second highest.

|   | Semantic ID construction | Search |  |  | Recommendation |  |  |
|---|--------------------------|--------|------|------|----------------|------|------|
|   |                          | All | All | Head | Torso |
| **Task-specific** | Search based | 0.072 | 0.026 | 0.090 | 0.070 |
|   | Rec. based | 0.004 | **0.062** | 0.170 | 0.035 |
| **Cross-task** | Separate | 0.028 | 0.027 | 0.090 | 0.035 |
|   | Concat | 0.051 | 0.031 | 0.090 | 0.035 |
|   | Fused<sub>CKA</sub> | **0.048** | 0.018 | 0.050 | 0.010 |
|   | Fused<sub>concat</sub> | 0.041 | <u>0.047</u> | 0.130 | 0.025 |
|   | Multi-task | <u>0.046</u> | **0.049** | 0.130 | 0.030 |
