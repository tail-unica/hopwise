t-SNE Embedding Visualization
=============================

This tutorial demonstrates how to visualize embeddings learned by hopwise models using t-SNE
(t-Distributed Stochastic Neighbor Embedding). This is particularly useful for understanding
how your model represents users, items, entities, and relations in the embedding space.

Prerequisites
-------------

Install the required dependencies:

.. code:: bash

    uv pip install hopwise[tsne]

This installs ``openTSNE`` and ``plotly`` for visualization.

Loading a Trained Model
-----------------------

First, load a trained model checkpoint:

.. code:: python

    import os
    import torch
    import numpy as np
    from openTSNE import TSNE
    import plotly.express as px

    from hopwise.utils import init_seed

    # Load checkpoint
    checkpoint_name = "TransE-Jan-23-2025_16-48-43.pth"
    checkpoint = torch.load(os.path.join("saved", checkpoint_name), weights_only=False)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])

    # Extract embeddings to CPU numpy arrays
    for weight in checkpoint["state_dict"].keys():
        checkpoint["state_dict"][weight] = checkpoint["state_dict"][weight].to(torch.device("cpu")).numpy()

Configuring t-SNE
-----------------

Configure the t-SNE algorithm:

.. code:: python

    tsne = TSNE(
        perplexity=30,
        n_jobs=8,
        initialization="random",
        metric="cosine",
        random_state=config["seed"],
        verbose=True,
    )

.. note::
    See the `openTSNE documentation <https://opentsne.readthedocs.io/en/stable/examples/02_advanced_usage/02_advanced_usage.html>`_
    for advanced configuration options.

Visualizing User Embeddings
---------------------------

.. code:: python

    user_weights = checkpoint["state_dict"]["user_embedding.weight"]
    tsne_embeddings_users = tsne.fit(user_weights)

    fig = px.scatter(
        x=tsne_embeddings_users[:, 0],
        y=tsne_embeddings_users[:, 1],
        color=list(range(len(tsne_embeddings_users))),
        labels={"x": "Dimension 1", "y": "Dimension 2", "color": "User ID"},
        title=f"{config['model']} User Embeddings",
        width=1024,
        height=1024,
        template="plotly_white",
    )
    fig.show()

Visualizing Entity Embeddings
-----------------------------

For knowledge-aware models, visualize entity embeddings:

.. code:: python

    entity_weights = checkpoint["state_dict"]["entity_embedding.weight"]
    tsne_embeddings_entities = tsne.fit(entity_weights)

    # Plot entities
    fig = px.scatter(
        x=tsne_embeddings_entities[:, 0],
        y=tsne_embeddings_entities[:, 1],
        color=list(range(len(tsne_embeddings_entities))),
        labels={"x": "Dimension 1", "y": "Dimension 2", "color": "Entity ID"},
        title=f"{config['model']} Entity Embeddings",
        width=1024,
        height=1024,
        template="plotly_white",
    )
    fig.show()

Combining Multiple Embedding Types
----------------------------------

Visualize different embedding types together:

.. code:: python

    import pandas as pd

    def combine_embeddings(**kwargs):
        embeddings_list = []
        identifiers_list = []

        for name, embs in kwargs.items():
            embeddings_list.append(embs)
            identifiers_list.extend([f"{name} {i}" for i in range(embs.shape[0])])

        embeddings = np.concatenate(embeddings_list, axis=0)

        df = pd.DataFrame({
            "x": embeddings[:, 0],
            "y": embeddings[:, 1],
            "type": [id.split(" ")[0] for id in identifiers_list],
            "identifier": identifiers_list,
        })

        fig = px.scatter(
            df, x="x", y="y", color="type",
            hover_data=["identifier"],
            title="Combined Embeddings Visualization",
            width=1024, height=1024,
            template="plotly_white",
        )
        fig.show()

    combine_embeddings(
        user=tsne_embeddings_users,
        entity=tsne_embeddings_entities,
        relation=tsne_embeddings_relations
    )

Full Example
------------

A complete Jupyter notebook example is available at:
``run_example/tSNE embedding visualisation.ipynb``