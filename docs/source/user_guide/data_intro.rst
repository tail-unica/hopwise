Data Module Introduction
=========================

**Hopwise is based on RecBole, so the data formatting is the same.**

Hopwise implements lots of popular recommender models, and also collects and releases 44 commonly-used published datasets.
You can freely download these datasets following our docs :doc:`datasets/dataset_download`.

For extensibility and reusability, Hopwise has a flexible and extensible data module.
Our data module designs an elegant data flow that transforms raw data
into the model input. Detailed as :doc:`datasets/data_flow`.
In order to characterize most forms of the input data
required by different recommendation tasks, Hopwise uses an input data format called :doc:`datasets/atomic_files`. All the input data should be
converted into `Atomic Files` format.
Besides, we design a data structure called :doc:`datasets/interaction` to provide a unified internal data representation for different
recommendation algorithms.

Hopwise supports both explicit feedback (labeled data) scenes and implicit feedback (unlabeled data) scenes. For explicit feedback scenes,
users can set the `LABEL_FIELD` in the config and Hopwise will train and test the model based on the label. For implicit feedback scenes, Hopwise will
regard all the observed interactions as positive samples and automatically select the negative samples from the unobserved interactions (which is known as negative sampling).
For more information about label setting,
please read :doc:`datasets/label_of_data`.

Here are the related docs for data module:

.. toctree::
   :maxdepth: 1

   datasets/dataset_download
   datasets/data_flow
   datasets/atomic_files
   datasets/interaction
   datasets/label_of_data