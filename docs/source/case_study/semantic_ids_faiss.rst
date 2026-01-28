Semantic IDs with FAISS
=======================

This tutorial explains how to generate semantic IDs using FAISS (Facebook AI Similarity Search)
for efficient nearest neighbor clustering. This approach uses hierarchical k-means clustering
to create discrete item representations.

Overview
--------

FAISS-based semantic ID generation works by:

1. Clustering item embeddings using hierarchical k-means
2. Assigning each item a path through the cluster hierarchy
3. Using the cluster path as a semantic ID

Prerequisites
-------------

Install FAISS:

.. code:: bash

    uv pip install faiss-cpu  # or faiss-gpu for GPU support

Generating Semantic IDs
-----------------------

.. code:: python

    import faiss
    import numpy as np

    def generate_faiss_semantic_ids(
        embeddings: np.ndarray,
        n_clusters_per_level: int = 256,
        n_levels: int = 3,
    ):
        """
        Generate hierarchical semantic IDs using FAISS k-means.

        Args:
            embeddings: Item embeddings of shape [n_items, embedding_dim]
            n_clusters_per_level: Number of clusters at each level
            n_levels: Depth of the hierarchy

        Returns:
            semantic_ids: Array of shape [n_items, n_levels]
        """
        n_items, dim = embeddings.shape
        semantic_ids = np.zeros((n_items, n_levels), dtype=np.int32)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        current_embeddings = embeddings.copy()

        for level in range(n_levels):
            # Perform k-means clustering
            kmeans = faiss.Kmeans(
                d=dim,
                k=n_clusters_per_level,
                niter=20,
                verbose=True,
                gpu=False,  # Set to True for GPU
            )
            kmeans.train(current_embeddings)

            # Assign items to clusters
            _, assignments = kmeans.index.search(current_embeddings, 1)
            semantic_ids[:, level] = assignments.flatten()

            # Compute residuals for next level
            centroids = kmeans.centroids[assignments.flatten()]
            current_embeddings = current_embeddings - centroids
            faiss.normalize_L2(current_embeddings)

        return semantic_ids

Example Usage
-------------

.. code:: python

    from sentence_transformers import SentenceTransformer

    # Load item descriptions and generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    item_descriptions = ["Action movie", "Comedy film", "Drama series", ...]
    embeddings = model.encode(item_descriptions)

    # Generate semantic IDs
    semantic_ids = generate_faiss_semantic_ids(
        embeddings,
        n_clusters_per_level=64,
        n_levels=4,
    )

    # semantic_ids[i] gives the hierarchical cluster path for item i
    print(f"Item 0 semantic ID: {semantic_ids[0]}")
    # Output: Item 0 semantic ID: [23, 45, 12, 7]

Comparison with RQ-VAE
----------------------

+------------------+---------------------------+---------------------------+
| Aspect           | FAISS                     | RQ-VAE                    |
+==================+===========================+===========================+
| Training         | No training required      | Requires training         |
+------------------+---------------------------+---------------------------+
| Flexibility      | Fixed clustering          | Learnable representations |
+------------------+---------------------------+---------------------------+
| Speed            | Very fast                 | Moderate                  |
+------------------+---------------------------+---------------------------+
| Quality          | Good for similar items    | Better semantic capture   |
+------------------+---------------------------+---------------------------+

Use Cases
---------

- **Fast prototyping**: When you need semantic IDs quickly without training
- **Large-scale datasets**: FAISS scales well to millions of items
- **Baseline comparison**: Compare against learned methods like RQ-VAE

See Also
--------

- :doc:`rq_vae_semantic_ids`: Learned semantic IDs with RQ-VAE
- `FAISS Documentation <https://github.com/facebookresearch/faiss>`_