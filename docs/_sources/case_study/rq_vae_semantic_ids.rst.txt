Semantic IDs with RQ-VAE
========================

This tutorial explains how to generate semantic IDs for items using Residual Quantization
Variational Autoencoders (RQ-VAE). Semantic IDs provide a hierarchical discrete representation
of items that can be used for efficient retrieval and recommendation.

.. note::
    This implementation is based on the work from:
    https://github.com/justinhangoebl/Semantic-ID-Generation

Overview
--------

RQ-VAE learns to compress item embeddings into a sequence of discrete codes (semantic IDs)
using residual quantization. Each layer of quantization captures increasingly fine-grained
information about the item.

**Key components:**

1. **Encoder**: Maps item features to a latent representation
2. **Quantization Layers**: Convert continuous latent vectors to discrete codes
3. **Decoder**: Reconstructs the original features from quantized representations

Usage
-----

The complete implementation is available at ``run_example/RQ_Vae_Semantic_IDs.py``.

.. code:: python

    from run_example.RQ_Vae_Semantic_IDs import RQ_VAE

    # Initialize the model
    model = RQ_VAE(
        input_dim=768,              # Dimension of input embeddings
        latent_dim=64,              # Latent space dimension
        hidden_dims=[256, 128],     # Hidden layer dimensions
        codebook_size=256,          # Number of codes per quantization layer
        n_quantization_layers=3,    # Number of residual quantization layers
        codebook_kmeans_init=True,  # Initialize codebooks with k-means
        commitment_weight=0.25,     # Commitment loss weight
    )

    # Initialize codebooks with k-means on your data
    model.kmeans_init_codebooks(item_embeddings)

    # Get semantic IDs for items
    output = model.get_semantic_ids(item_embeddings)
    semantic_ids = output.sem_ids  # Shape: [num_items, n_quantization_layers]

Training
--------

The RQ-VAE is trained with a combination of:

- **Reconstruction loss**: Ensures the decoder can reconstruct the original embeddings
- **Quantization loss**: Encourages the encoder output to match codebook entries
- **Commitment loss**: Prevents the encoder from fluctuating between codebook entries

.. code:: python

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        for batch in dataloader:
            output = model(batch)

            # Total loss = reconstruction + quantization losses
            loss = output.reconstruction_loss + output.quantize_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

Integration with Hopwise
------------------------

Semantic IDs can be used with sequential recommendation models to represent items
as sequences of discrete tokens rather than continuous embeddings. This enables:

- More efficient storage and retrieval
- Better interpretability of item representations
- Compatibility with language model-based recommendation

Full Example
------------

See the complete script at: ``run_example/RQ_Vae_Semantic_IDs.py``