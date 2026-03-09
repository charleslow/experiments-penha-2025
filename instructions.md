# Replicate Penha 2025

Semantic IDs for Joint Generative Search and Recommendation
Paper: https://arxiv.org/abs/2508.10478
Paper summary: https://charleslow.github.io/notebook/book/papers/penha_2025.html

The goal of this repo is to perform experiments to validate the results from penha 2025. We do not need to replicate all results, just the important ones.

The main goal is to show that:
1. RQVAE is poorer than RQ-kmeans when doing semantic ID recommendation
2. Semantic IDs trained on query-specific embeddings do poorly on recommendation and vice versa
3. Using their multi-task method creates semantic IDs that do decently on either task

Important: we need to generate synthetic queries using gemini-2.0-flash. Since we do not have an API key, use a reasonable local model to do so instead. Note that a specific prompt was used in the paper to avoid queries from being too similar to the target movie:
```
Your task is to return a list with 10 queries for a given
movie (title of the movie, year and description and tags) After generating the initial
set of queries, you should also generate a list of the same size with paraphrased of the
first queries. The paraphrased queries should be similar to the original queries, but with
different words, structure and slight variations in the meaning. The queries should be
realistic things that a user would ask to find the movie. The queries should be diverse and
cover different aspects of the movie. The queries should not include the title of the movie,
but be broader descriptions of the movie and its content. The queries should also contain
broad topics, themes and genres of the movie. Movie: {METADATA}
```
