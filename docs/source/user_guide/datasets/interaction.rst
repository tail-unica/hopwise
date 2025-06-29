Interaction
================

:class:`~hopwise.data.interaction.Interaction` is the internal data structural that is loaded from :class:`DataLoader`, and fed into the recommendation algorithms.

It is implemented as a new abstract data type based on :class:`python.dict`. The keys correspond to features from input, which can be conveniently referenced with feature names when writing the recommendation algorithms; and the values correspond to tensors (implemented by :class:`torch.Tensor`), which will be used for the update and computation in learning algorithms. Specially, the value entry for a specific key stores all the corresponding tensor data in a batch or mini-batch.

With such a data structure, our library provides a friendly interface to write the recommendation algorithms in a batch-based mode. For example, we can read all the user embeddings and item embeddings from an instantiated :class:`~hopwise.data.interaction.Interaction` object ``inter`` simply based on the feature names:

.. code:: python

    user_vec = inter['UserID']
    item_vec = inter['ItemID']

The contents of an :class:`~hopwise.data.interaction.Interaction` are decided by the loaded fields.
However, it should be noted that there can be some features generated by :class:`DataLoader`, e.g. if one model has ``input_type = InputType.PAIRWISE``, then each item feature has a corresponding negative item feature, whose keys are begin with arg ``NEG_PREFIX``.

Besides, the value components are implemented based on :class:`torch.Tensor`. We wrap many functions of PyTorch to develop a GRU-oriented data structure, which can support batch-based mechanism (e.g., copying a batch of data to GPU). In specific, we summarize the important functions as follows:

============================         ==================================================================
Function                             Description
============================         ==================================================================
to(device)                           transfer all tensors to :class:`torch.device`
cpu                                  transfer all tensors to CPU
numpy                                transfer all tensors to :class:`numpy.ndarray`
repeat                               repeats each tensor along the batch size dimension
repeat interleave                    repeat elements of a tensor, similar to repeat interleave
update                               update this object with another Interaction, similar to update
============================         ==================================================================
