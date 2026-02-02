# Table 1: Fine-tuning the embeddings for Semantic ID construction

Fine-tuning the embeddings—used for Semantic ID construction—for search and recommendation is effective in a joint generative model. However, choosing one of the fine-tuned embedding spaces comes at the expense of the other task effectiveness. **Bold** indicates highest effectiveness while the superscripts denote statistical significance using paired t-tests and Bonferroni correction.

|   | Embedding space | Search R@30 (± std.) | Recommendation R@30 (± std.) |
|---|-----------------|----------------------|------------------------------|
| 1 | Content-based (e.g. DSI [21], TIGER [18]) | 0.013 (±0.009) | 0.023 (±0.017) |
| 2 | Search based (e.g. RIPOR [26]) | **0.072 (±0.028)**<sup>13</sup> | 0.026 (±0.017) |
| 3 | Rec. based (e.g. TokenRec [16]) | 0.004 (±0.001) | **0.062 (±0.015)**<sup>12</sup> |
